package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/unixpickle/model3d/model3d"
)

// A VectorFlag is a flag.Value that parses comma-delimited
// 3D vectors, e.g. "3.0, 2, -1".
type VectorFlag struct {
	Value model3d.Coord3D
}

func (v *VectorFlag) String() string {
	var parts [3]string
	for i, x := range v.Value.Array() {
		parts[i] = strconv.FormatFloat(x, 'f', -1, 64)
	}
	return strings.Join(parts[:], ",")
}

func (v *VectorFlag) Set(s string) error {
	parts := strings.Split(s, ",")
	if len(parts) != 3 {
		return fmt.Errorf("vector does not have exactly two commas: %s", s)
	}
	var res [3]float64
	for i, x := range parts {
		x = strings.TrimSpace(x)
		f, err := strconv.ParseFloat(x, 64)
		if err != nil {
			return fmt.Errorf("invalid component '%s' in vector '%s': %s", x, s, err.Error())
		}
		res[i] = f
	}
	v.Value = model3d.NewCoord3DArray(res)
	return nil
}
