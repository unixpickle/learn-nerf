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
	var lightBrightness float64
	var red float64
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

	mesh = mesh.Translate(mesh.Min().Mid(mesh.Max()).Scale(-1))
	m := mesh.Max()
	size := math.Max(math.Max(m.X, m.Y), m.Z)
	mesh = mesh.Scale(1 / size)

	collider := model3d.MeshToCollider(mesh)
	object := render3d.Objectify(collider, func(c model3d.Coord3D, rc model3d.RayCollision) render3d.Color {
		return render3d.NewColorRGB(red, green, blue)
	})
	center := object.Min().Mid(object.Max())

	log.Println("Writing metadata...")
	globalMetadataPath := filepath.Join(outputDir, "metadata.json")
	f, err := os.Create(globalMetadataPath)
	essentials.Must(err)
	globalMetadata := map[string]interface{}{"min": object.Min().Array(), "max": object.Max().Array()}
	essentials.Must(json.NewEncoder(f).Encode(globalMetadata))
	f.Close()

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
		camera := render3d.DirectionalCamera(object, direction, fov*math.Pi/180)
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
			"origin": camera.Origin.Array(),
			"x":      camera.ScreenX.Array(),
			"y":      camera.ScreenY.Array(),
			"z":      camera.ScreenX.Cross(camera.ScreenY).Normalize().Array(),
			"x_fov":  camera.FieldOfView,
			"y_fov":  camera.FieldOfView,
		}
		f, err := os.Create(metaPath)
		essentials.Must(err)
		essentials.Must(json.NewEncoder(f).Encode(metadata))
		essentials.Must(f.Close())
	}
}
