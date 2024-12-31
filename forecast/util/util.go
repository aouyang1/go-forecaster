package util

func IndentExpand(indent string, growth int) string {
	indentByte := []byte(indent)
	out := make([]byte, 0, len(indent)*growth)
	for i := 0; i < growth; i++ {
		out = append(out, indentByte...)
	}
	return string(out)
}
