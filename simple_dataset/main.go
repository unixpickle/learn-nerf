// Command simple_dataset creates a NeRF dataset from a single-color STL file.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
)

func main() {
	var fov float64
	var resolution int
	var numImages int
	var numLights int
	var red float64
	var lightBrightness float64
	var green float64
	var blue float64
	flag.Float64Var(&fov, "fov", 60.0, "field of view in degrees")
	flag.IntVar(&resolution, "resolution", 800, "side length of images to render")
	flag.IntVar(&numImages, "images", 100, "number of images to render")
	flag.IntVar(&numLights, "num-lights", 5, "number of lights to put into the scene")
	flag.Float64Var(&lightBrightness, "light-brightness", 0.5, "brightness of lights")
	flag.Float64Var(&red, "red", 0.8, "red color component")
	flag.Float64Var(&green, "green", 0.8, "green color component")
	flag.Float64Var(&blue, "blue", 0.0, "blue color component")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: simple_dataset [flags] <input.stl> <output-dir>")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Flags:")
		flag.PrintDefaults()
		os.Exit(1)
	}
	flag.Parse()
	if len(flag.Args()) != 2 {
		flag.Usage()
	}
	inputPath := flag.Args()[0]
	outputDir := flag.Args()[1]
	if stats, err := os.Stat(outputDir); err == nil && !stats.IsDir() {
		essentials.Die("output directory already exists: " + outputDir)
	} else if os.IsNotExist(err) {
		essentials.Must(os.MkdirAll(outputDir, 0755))
	}

	log.Println("Loading mesh...")
	r, err := os.Open(inputPath)
	essentials.Must(err)
	triangles, err := model3d.ReadSTL(r)
	essentials.Must(r.Close())
	essentials.Must(err)
	mesh := model3d.NewMeshTriangles(triangles)
	collider := model3d.MeshToCollider(mesh)
	object := render3d.Objectify(collider, func(c model3d.Coord3D, rc model3d.RayCollision) render3d.Color {
		return render3d.NewColorRGB(red, green, blue)
	})
	center := object.Min().Mid(object.Max())

	log.Println("Creating random lights...")
	lights := []*render3d.PointLight{}
	for i := 0; i < numLights; i++ {
		direction := model3d.NewCoord3DRandUnit()
		lights = append(lights, &render3d.PointLight{
			Origin: center.Add(direction.Scale(1000)),
			Color:  render3d.NewColor(lightBrightness),
		})
	}

	for i := 0; i < numImages; i++ {
		log.Printf("Rendering imade %d/%d...", i+1, numImages)
		direction := model3d.NewCoord3DRandUnit()
		camera := directionalCamera(object, direction, fov*math.Pi/180)
		caster := &render3d.RayCaster{
			Camera: camera,
			Lights: lights,
		}
		viewImage := render3d.NewImage(resolution, resolution)
		caster.Render(viewImage, object)

		imagePath := filepath.Join(outputDir, fmt.Sprintf("%04d.png", i))
		essentials.Must(viewImage.Save(imagePath))

		metaPath := filepath.Join(outputDir, fmt.Sprintf("%04d.json", i))
		metadata := map[string]interface{}{
			"origin": camera.Origin,
			"x":      camera.ScreenX,
			"y":      camera.ScreenY,
			"z":      camera.ScreenX.Cross(camera.ScreenY).Normalize(),
			"fov":    camera.FieldOfView,
		}
		f, err := os.Create(metaPath)
		essentials.Must(err)
		essentials.Must(json.NewEncoder(f).Encode(metadata))
		essentials.Must(f.Close())
	}
}

// directionalCamera figures out where to move a camera in
// the given unit direction to capture the bounding box of
// an object.
func directionalCamera(object render3d.Object, direction model3d.Coord3D, fov float64) *render3d.Camera {
	min, max := object.Min(), object.Max()
	baseline := min.Dist(max)
	center := min.Mid(max)

	margin := 0.05
	minDist := baseline * 1e-4
	maxDist := baseline * 1e4
	for i := 0; i < 32; i++ {
		d := (minDist + maxDist) / 2
		cam := render3d.NewCameraAt(center.Add(direction.Scale(d)), center, fov)
		uncaster := cam.Uncaster(1, 1)
		contained := true
		for _, x := range []float64{min.X, max.X} {
			for _, y := range []float64{min.Y, max.Y} {
				for _, z := range []float64{min.Z, max.Z} {
					sx, sy := uncaster(model3d.XYZ(x, y, z))
					if sx < margin || sy < margin || sx >= 1-margin || sy >= 1-margin {
						contained = false
					}
				}
			}
		}
		if contained {
			maxDist = d
		} else {
			minDist = d
		}
	}

	return render3d.NewCameraAt(center.Add(direction.Scale(maxDist)), center, fov)
}
