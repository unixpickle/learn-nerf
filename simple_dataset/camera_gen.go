package main

import (
	"math"

	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
)

type CameraGen interface {
	Camera(i, total int) *render3d.Camera
}

type RandomCameraGen struct {
	Object render3d.Object
	Fov    float64
}

func (r *RandomCameraGen) Camera(i, total int) *render3d.Camera {
	direction := model3d.NewCoord3DRandUnit()
	return render3d.DirectionalCamera(r.Object, direction, r.Fov*math.Pi/180)
}

type RotatingCameraGen struct {
	Object render3d.Object
	Fov    float64
	Axis   model3d.Coord3D
	Offset model3d.Coord3D

	total         int
	furthestScale float64
}

func (r *RotatingCameraGen) Camera(i, total int) *render3d.Camera {
	if r.total != total {
		r.updateCache(total)
	}
	dir := r.direction(i, total)
	center := r.Object.Min().Mid(r.Object.Max())
	cam := render3d.NewCameraAt(center.Add(dir.Scale(r.furthestScale)), center, r.Fov)
	return cam
}

func (r *RotatingCameraGen) updateCache(total int) {
	r.total = total
	scale := 0.0
	for i := 0; i < total; i++ {
		cam := render3d.DirectionalCamera(r.Object, r.direction(i, total), r.Fov)
		s := cam.Origin.Dist(r.Object.Min().Mid(r.Object.Max()))
		if s > scale {
			scale = s
		}
	}
	r.furthestScale = scale
}

func (r *RotatingCameraGen) direction(i, total int) model3d.Coord3D {
	theta := math.Pi * 2 * float64(i) / float64(total)
	rotation := model3d.Rotation(r.Axis, theta)
	return rotation.Apply(r.Offset)
}
