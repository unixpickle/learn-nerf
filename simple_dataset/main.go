// Command simple_dataset creates a NeRF dataset from a single-color STL file.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
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
	var noImages bool
	var seed int64
	color := VectorFlag{Value: model3d.XYZ(0.8, 0.8, 0.0)}

	flag.Float64Var(&fov, "fov", 60.0, "field of view in degrees")
	flag.IntVar(&resolution, "resolution", 800, "side length of images to render")
	flag.IntVar(&numImages, "images", 100, "number of images to render")
	flag.IntVar(&numLights, "num-lights", 5, "number of lights to put into the scene")
	flag.Float64Var(&lightBrightness, "light-brightness", 0.5, "brightness of lights")
	flag.BoolVar(&noImages, "no-images", false, "only save json files, not renderings")
	flag.Int64Var(&seed, "seed", 0, "seed for Go's random number generation")
	flag.Var(&color, "color", "color of the model, as 'r,g,b'")

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

	rand.Seed(seed)

	outputDir := flag.Args()[1]
	log.Printf("Creating output directory: %s...", outputDir)
	if stats, err := os.Stat(outputDir); err == nil && !stats.IsDir() {
		essentials.Die("output directory already exists: " + outputDir)
	} else if os.IsNotExist(err) {
		essentials.Must(os.MkdirAll(outputDir, 0755))
	}

	log.Println("Loading model...")
	inputPath := flag.Args()[0]
	object := ReadObject(inputPath, color.Value)

	log.Println("Writing metadata...")
	WriteGlobalMetadata(outputDir, object)

	log.Println("Creating random lights...")
	lights := RandomLights(object, numLights, lightBrightness)

	for i := 0; i < numImages; i++ {
		log.Printf("Rendering imade %d/%d...", i+1, numImages)
		camera := RandomCamera(object, fov)

		if !noImages {
			caster := &render3d.RayCaster{
				Camera: camera,
				Lights: lights,
			}
			viewImage := render3d.NewImage(resolution, resolution)
			caster.Render(viewImage, object)

			imagePath := filepath.Join(outputDir, fmt.Sprintf("%04d.png", i))
			essentials.Must(viewImage.Save(imagePath))
		}

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

func ReadObject(path string, color model3d.Coord3D) render3d.Object {
	r, err := os.Open(path)
	essentials.Must(err)
	defer r.Close()

	triangles, err := model3d.ReadSTL(r)
	essentials.Must(err)
	mesh := normalizeMesh(model3d.NewMeshTriangles(triangles))

	collider := model3d.MeshToCollider(mesh)
	return render3d.Objectify(
		collider,
		func(c model3d.Coord3D, rc model3d.RayCollision) render3d.Color {
			return render3d.NewColorRGB(color.X, color.Y, color.Z)
		},
	)
}

func normalizeMesh(mesh *model3d.Mesh) *model3d.Mesh {
	mesh = mesh.Translate(mesh.Min().Mid(mesh.Max()).Scale(-1))
	m := mesh.Max()
	size := math.Max(math.Max(m.X, m.Y), m.Z)
	return mesh.Scale(1 / size)
}

func WriteGlobalMetadata(outputDir string, object render3d.Object) {
	globalMetadataPath := filepath.Join(outputDir, "metadata.json")
	f, err := os.Create(globalMetadataPath)
	essentials.Must(err)
	defer f.Close()
	globalMetadata := map[string]interface{}{
		"min": object.Min().Array(),
		"max": object.Max().Array(),
	}
	essentials.Must(json.NewEncoder(f).Encode(globalMetadata))
}

func RandomLights(object render3d.Object, n int, brightness float64) []*render3d.PointLight {
	center := object.Min().Mid(object.Max())
	lights := make([]*render3d.PointLight, n)
	for i := 0; i < n; i++ {
		direction := model3d.NewCoord3DRandUnit()
		lights[i] = &render3d.PointLight{
			Origin: center.Add(direction.Scale(1000)),
			Color:  render3d.NewColor(brightness),
		}
	}
	return lights
}

func RandomCamera(object render3d.Object, fov float64) *render3d.Camera {
	direction := model3d.NewCoord3DRandUnit()
	return render3d.DirectionalCamera(object, direction, fov*math.Pi/180)
}
