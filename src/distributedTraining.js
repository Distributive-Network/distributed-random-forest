import * as dcp from 'dcp-client';
import {
  DecisionTreeClassifier as DTClassifier,
  DecisionTreeRegression as DTRegression,
} from 'ml-cart';
import {
  Matrix,
  MatrixColumnSelectionView,
} from 'ml-matrix';

import * as Utils from './utils';

/**
   * Train a random forest with the given training set and labels in a distribted manner.
   * Training is separated into "slices", each of which handles training one or more trees.
   * Distributed computing is handled using DCP. A valid DCP key must be configured on the
   * machine running this code for distributed training to work.
   * @param {Matrix|Array} trainingSet
   * @param {Array} trainingValues
   */
export function distributedTrain(trainingSet, trainingValues) {
  // TODO: add DCP args

  // TODO: import DCP
  /**
   * TODO: seed will be set in individual slices
   * How will we handle reproducability? Maybe use the first seed to set all subsequent seeds for slices?
   */
  let currentSeed = this.seed;

  trainingSet = Matrix.checkMatrix(trainingSet);

  this.maxFeatures = this.maxFeatures || trainingSet.columns;
  this.numberFeatures = trainingSet.columns;
  this.numberSamples = trainingSet.rows;

  if (Utils.checkFloat(this.maxFeatures)) {
    this.n = Math.floor(trainingSet.columns * this.maxFeatures);
  } else if (Number.isInteger(this.maxFeatures)) {
    if (this.maxFeatures > trainingSet.columns) {
      throw new RangeError(
        `The maxFeatures parameter should be less than ${trainingSet.columns}`,
      );
    } else {
      this.n = this.maxFeatures;
    }
  } else {
    throw new RangeError(
      `Cannot process the maxFeatures parameter ${this.maxFeatures}`,
    );
  }

  if (this.maxSamples) {
    if (this.maxSamples < 0) {
      throw new RangeError(`Please choose a positive value for maxSamples`);
    } else {
      if (Utils.isFloat(this.maxSamples)) {
        if (this.maxSamples > 1.0) {
          throw new RangeError(
            'Please choose either a float value between 0 and 1 or a positive integer for maxSamples',
          );
        } else {
          this.numberSamples = Math.floor(trainingSet.rows * this.maxSamples);
        }
      } else if (Number.isInteger(this.maxSamples)) {
        if (this.maxSamples > trainingSet.rows) {
          throw new RangeError(
            `The maxSamples parameter should be less than ${trainingSet.rows}`,
          );
        } else {
          this.numberSamples = this.maxSamples;
        }
      }
    }
  }

  if (this.maxSamples) {
    if (trainingSet.rows !== this.numberSamples) {
      let tmp = new Matrix(this.numberSamples, trainingSet.columns);
      for (let j = 0; j < this.numberSamples; j++) {
        tmp.removeRow(0);
      }
      for (let i = 0; i < this.numberSamples; i++) {
        tmp.addRow(trainingSet.getRow(i));
      }
      trainingSet = tmp;

      trainingValues = trainingValues.slice(0, this.numberSamples);
    }
  }

  // TODO: validate DCP inputs (?)

  // TODO: most of this will go inside a work function.

  let Estimator;
  if (this.isClassifier) {
    Estimator = DTClassifier;
  } else {
    Estimator = DTRegression;
  }

  this.estimators = new Array(this.nEstimators);
  this.indexes = new Array(this.nEstimators);

  let oobResults = new Array(this.nEstimators);

  // TODO: this is fundamental loop for DCP to parallelize
  for (let i = 0; i < this.nEstimators; ++i) {
    let res = this.useSampleBagging
      ? Utils.examplesBaggingWithReplacement(
          trainingSet,
          trainingValues,
          currentSeed,
        )
      : {
          X: trainingSet,
          y: trainingValues,
          seed: currentSeed,
          Xoob: undefined,
          yoob: [],
          ioob: [],
        };
    let X = res.X;
    let y = res.y;
    currentSeed = res.seed;
    let { Xoob, ioob } = res;

    // Other implementations of random forests apply feature bagging at every split during tree generation.
    // So I think it would be better to implement it at the CART level, not here.

    res = Utils.featureBagging(X, this.n, this.replacement, currentSeed);
    X = res.X;
    currentSeed = res.seed;

    // setting these will need to be done outside the work fn (although a work fn may train multiple estimators)
    this.indexes[i] = res.usedIndex;
    this.estimators[i] = new Estimator(this.treeOptions);
    this.estimators[i].train(X, y);

    // TODO: fix OOB issue (see issues section of original repo)

    if (!this.noOOB && this.useSampleBagging) {
      let xoob = new MatrixColumnSelectionView(Xoob, this.indexes[i]);
      oobResults[i] = {
        index: ioob,
        predicted: this.estimators[i].predict(xoob),
      };
    }
  }
  if (!this.noOOB && this.useSampleBagging && oobResults.length > 0) {
    this.oobResults = Utils.collectOOB(
      oobResults,
      trainingValues,
      this.selection.bind(this),
    );
  }
}