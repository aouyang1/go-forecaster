package feature

type Feature interface {
	String() string
	Get(string) (string, bool)
}
