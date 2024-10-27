package models

import (
	"errors"
)

var (
	ErrNoOptions          = errors.New("no initialized model options")
	ErrTargetLenMismatch  = errors.New("target length does not match target rows")
	ErrNoTrainingMatrix   = errors.New("no training matrix")
	ErrNoTargetMatrix     = errors.New("no target matrix")
	ErrNoDesignMatrix     = errors.New("no design matrix for inference")
	ErrFeatureLenMismatch = errors.New("number of features does not match number of model coefficients")
)
