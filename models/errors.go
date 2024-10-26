package models

import (
	"errors"
)

var (
	ErrNoOptions                 = errors.New("no initialized model options")
	ErrTargetLenMismatch         = errors.New("target length does not match target rows")
	ErrNoTrainingArray           = errors.New("no training array")
	ErrNoTargetArray             = errors.New("no target array")
	ErrNoDesignMatrix            = errors.New("no design matrix for inference")
	ErrFeatureLenMismatch        = errors.New("number of features does not match number of model coefficients")
	ErrObsYSizeMismatch          = errors.New("observation and y matrix have different number of features")
	ErrWarmStartBetaSize         = errors.New("warm start beta does not have the same dimensions as X")
	ErrInsufficientSamples       = errors.New("insufficient samples for the determined folds")
	ErrInconsistentSampleLengths = errors.New("features or targets do not have the same number of samples")
)
