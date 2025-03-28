package util

func IndentExpand(indent string, growth int) string {
	indentByte := []byte(indent)
	out := make([]byte, 0, len(indent)*growth)
	for i := 0; i < growth; i++ {
		out = append(out, indentByte...)
	}
	return string(out)
}

func SliceMap(arr []float64, lambda func(float64) float64) []float64 {
	for i, v := range arr {
		arr[i] = lambda(v)
	}
	return arr
}
