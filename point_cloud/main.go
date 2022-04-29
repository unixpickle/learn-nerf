// Command point_cloud reconstructs a point cloud from a dataset with added
// depth images.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image/color"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
	"github.com/unixpickle/model3d/toolbox3d"
)

func main() {
	var maxDepth float64
	var thickness float64
	var delta float64
	var maxPoints int
	var dataDir string
	var outputPath string
	flag.Float64Var(&maxDepth, "max-depth", 5.0, "maximum depth value corresponding to white pixel")
	flag.Float64Var(&thickness, "thickness", 0.02, "radius of each point")
	flag.Float64Var(&delta, "delta", 0.02, "marching cubes delta")
	flag.IntVar(&maxPoints, "max-points", 50000, "maximum points to sample")
	flag.StringVar(&dataDir, "data-dir", "", "data directory")
	flag.StringVar(&outputPath, "output-path", "", "output zipped material OBJ path")
	flag.Parse()
	if dataDir == "" || outputPath == "" {
		essentials.Die("Must specify -data-dir and -output-path")
	}

	log.Println("Computing points...")
	points := []model3d.Coord3D{}
	colors := []render3d.Color{}
	for i := 0; true; i++ {
		metadataPath := filepath.Join(dataDir, fmt.Sprintf("%05d.json", i))
		depthPath := filepath.Join(dataDir, fmt.Sprintf("%05d_depth.png", i))
		colorPath := filepath.Join(dataDir, fmt.Sprintf("%05d.png", i))

		if _, err := os.Stat(metadataPath); os.IsNotExist(err) {
			break
		}
		var metadata struct {
			Origin [3]float64 `json:"origin"`
			XFov   float64    `json:"x_fov"`
			YFov   float64    `json:"y_fov"`
			X      [3]float64 `json:"x"`
			Y      [3]float64 `json:"y"`
			Z      [3]float64 `json:"z"`
		}
		f, err := os.Open(metadataPath)
		essentials.Must(err)
		err = json.NewDecoder(f).Decode(&metadata)
		f.Close()
		essentials.Must(err)

		origin := model3d.NewCoord3DArray(metadata.Origin)
		xAxis := model3d.NewCoord3DArray(metadata.X).Scale(math.Tan(metadata.XFov / 2))
		yAxis := model3d.NewCoord3DArray(metadata.Y).Scale(math.Tan(metadata.YFov / 2))
		zAxis := model3d.NewCoord3DArray(metadata.Z)

		err = ReadRGBD(depthPath, colorPath, func(x, y float64, depth uint16, c color.Color) {
			zDist := (float64(depth) / 0xffff) * maxDepth
			direction := zAxis.Add(xAxis.Scale(x)).Add(yAxis.Scale(y)).Normalize()
			scale := zDist / direction.Dot(zAxis)
			coord := origin.Add(direction.Scale(scale))
			points = append(points, coord)
			r, g, b, _ := color.RGBAModel.Convert(c).RGBA()
			colors = append(colors, render3d.NewColorRGB(float64(r)/0xffff, float64(g)/0xffff, float64(b)/0xffff))
		})
		essentials.Must(err)
	}

	if len(points) > maxPoints {
		log.Printf("Found %d points. Reducing to %d...", len(points), maxPoints)
		rand.Shuffle(len(points), func(i, j int) {
			points[i], points[j] = points[j], points[i]
			colors[i], colors[j] = colors[j], colors[i]
		})
		points = points[:maxPoints]
		colors = colors[:maxPoints]
	} else {
		log.Printf("Using all %d points.", len(points))
	}

	log.Println("Constructing solid and color function...")
	min := points[0]
	max := points[0]
	for _, p := range points {
		min = min.Min(p)
		max = max.Max(p)
	}
	tree := model3d.NewCoordTree(points)
	solid := model3d.CheckedFuncSolid(
		min,
		max,
		func(c model3d.Coord3D) bool {
			return tree.Dist(c) < thickness
		},
	)
	coordToColor := map[model3d.Coord3D]render3d.Color{}
	for i, c := range points {
		coordToColor[c] = colors[i]
	}
	colorFunc := toolbox3d.CoordColorFunc(func(c model3d.Coord3D) render3d.Color {
		return coordToColor[tree.NearestNeighbor(c)]
	})

	log.Println("Creating mesh...")
	mesh := model3d.MarchingCubesSearch(solid, delta, 8)

	log.Println("Saving mesh...")
	mesh.SaveMaterialOBJ(outputPath, colorFunc.TriangleColor)
}

func ReadRGBD(depthPath, colorPath string, cb func(x, y float64, depth uint16, c color.Color)) error {
	f, err := os.Open(depthPath)
	if err != nil {
		return err
	}
	depthImg, err := png.Decode(f)
	f.Close()
	if err != nil {
		return err
	}

	f, err = os.Open(colorPath)
	if err != nil {
		return err
	}
	colorImg, err := png.Decode(f)
	f.Close()
	if err != nil {
		return err
	}

	b := depthImg.Bounds()
	if b != colorImg.Bounds() {
		return errors.New("mismatched size of RGB and depth images")
	}

	for y := 0; y < b.Dy(); y++ {
		yFrac := 2*float64(y)/float64(b.Dy()-1) - 1
		for x := 0; x < b.Dx(); x++ {
			xFrac := 2*float64(x)/float64(b.Dx()-1) - 1

			depth, _, _, _ := color.Gray16Model.Convert(depthImg.At(x, y)).RGBA()
			if depth == 0xffff {
				continue
			}
			c := colorImg.At(x, y)
			cb(xFrac, yFrac, uint16(depth), c)
		}
	}

	return nil
}
