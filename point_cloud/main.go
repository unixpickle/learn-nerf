// Command point_cloud reconstructs a point cloud from a dataset with added
// depth images.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image/color"
	"image/png"
	"log"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
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
	flag.StringVar(&outputPath, "output-path", "", "output STL path")
	flag.Parse()
	if dataDir == "" || outputPath == "" {
		essentials.Die("Must specify -data-dir and -output-path")
	}

	log.Println("Computing points...")
	points := []model3d.Coord3D{}
	for i := 0; true; i++ {
		metadataPath := filepath.Join(dataDir, fmt.Sprintf("%05d.json", i))
		imgPath := filepath.Join(dataDir, fmt.Sprintf("%05d_depth.png", i))

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
		xAxis := model3d.NewCoord3DArray(metadata.X)
		yAxis := model3d.NewCoord3DArray(metadata.Y)
		zAxis := model3d.NewCoord3DArray(metadata.Z)

		f, err = os.Open(imgPath)
		essentials.Must(err)
		img, err := png.Decode(f)
		f.Close()
		essentials.Must(err)
		for y := 0; y < img.Bounds().Dy(); y++ {
			yFrac := 2*float64(y)/float64(img.Bounds().Dy()-1) - 1
			for x := 0; x < img.Bounds().Dx(); x++ {
				xFrac := 2*float64(x)/float64(img.Bounds().Dx()-1) - 1
				brightness, _, _, _ := color.Gray16Model.Convert(img.At(x, y)).RGBA()
				if brightness == 0xffff {
					continue
				}
				zFrac := (float64(brightness) / 0xffff) * maxDepth
				coord := origin.Add(zAxis.Scale(zFrac)).Add(xAxis.Scale(xFrac)).Add(yAxis.Scale(yFrac))
				points = append(points, coord)
			}
		}
	}

	if len(points) > maxPoints {
		log.Printf("Found %d points. Reducing to %d...", len(points), maxPoints)
		rand.Shuffle(len(points), func(i, j int) {
			points[i], points[j] = points[j], points[i]
		})
		points = points[:maxPoints]
	} else {
		log.Printf("Using all %d points.", len(points))
	}

	log.Println("Constructing solid...")
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

	log.Println("Creating mesh...")
	mesh := model3d.MarchingCubesSearch(solid, delta, 8)

	log.Println("Saving mesh...")
	mesh.SaveGroupedSTL(outputPath)
}
