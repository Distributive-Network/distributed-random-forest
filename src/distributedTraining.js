import {
  DecisionTreeClassifier as DTClassifier,
  DecisionTreeRegression as DTRegression,
} from 'ml-cart';
import {
  MatrixColumnSelectionView,
} from 'ml-matrix';

import * as Utils from './utils.js';

export async function workFunction(sliceInput, jobArgs, trainingSet, trainingValues) {
  // extract info from slice arguments
  let [currentSeed, randomForestModelParams] = sliceInput;

  let Estimator;
  if (jobArgs.isClassifier) {
    Estimator = DTClassifier;
  } else {
    Estimator = DTRegression;
  }

  progress(0);

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

    // TODO: setting these will need to be done outside the work fn (although a work fn may train multiple estimators)
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
}