import * as dcpClient from 'dcp-client';
import {
  DecisionTreeClassifier as DTClassifier,
  DecisionTreeRegression as DTRegression,
} from 'ml-cart';
import {
  Matrix,
  MatrixColumnSelectionView,
  MatrixTransposeView,
  WrapperMatrix2D,
} from 'ml-matrix';

import * as Utils from './utils';

let dcpInitializer;

/**
 * @class RandomForestBase
 */
export class RandomForestBase {
  /**
   * Create a new base random forest for a classifier or regression model.
   * @constructor
   * @param {object} options
   * @param {number|String} [options.maxFeatures] - the number of features used on each estimator.
   *        * if is an integer it selects maxFeatures elements over the sample features.
   *        * if is a float between (0, 1), it takes the percentage of features.
   * @param {boolean} [options.replacement] - use replacement over the sample features.
   * @param {number} [options.seed] - seed for feature and samples selection, must be a 32-bit integer.
   * @param {number} [options.nEstimators] - number of estimator to use.
   * @param {object} [options.treeOptions] - options for the tree classifier, see [ml-cart]{@link https://mljs.github.io/decision-tree-cart/}
   * @param {boolean} [options.isClassifier] - boolean to check if is a classifier or regression model (used by subclasses).
   * @param {boolean} [options.useSampleBagging] - use bagging over training samples.
   * @param {boolean} [options.noOOB] - don't calculate Out-Of-Bag predictions.
   * @param {object} model - for load purposes.
   */
  constructor(options, model) {
    if (options === true) {
      this.replacement = model.replacement;
      this.maxFeatures = model.maxFeatures;
      this.nEstimators = model.nEstimators;
      this.treeOptions = model.treeOptions;
      this.isClassifier = model.isClassifier;
      this.seed = model.seed;
      this.n = model.n;
      this.indexes = model.indexes;
      this.useSampleBagging = model.useSampleBagging;
      this.noOOB = true;
      this.maxSamples = model.maxSamples;

      let Estimator = this.isClassifier ? DTClassifier : DTRegression;
      this.estimators = model.estimators.map((est) => Estimator.load(est));
    } else {
      this.replacement = options.replacement;
      this.maxFeatures = options.maxFeatures;
      this.nEstimators = options.nEstimators;
      this.treeOptions = options.treeOptions;
      this.isClassifier = options.isClassifier;
      this.seed = options.seed;
      this.useSampleBagging = options.useSampleBagging;
      this.noOOB = options.noOOB;
      this.maxSamples = options.maxSamples;
    }
  }

  /**
   * Train the decision tree with the given training set and labels.
   * @param {Matrix|Array} trainingSet
   * @param {Array} trainingValues
   */
  train(trainingSet, trainingValues) {
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

    let Estimator;
    if (this.isClassifier) {
      Estimator = DTClassifier;
    } else {
      Estimator = DTRegression;
    }

    this.estimators = new Array(this.nEstimators);
    this.indexes = new Array(this.nEstimators);

    let oobResults = new Array(this.nEstimators);

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

      this.indexes[i] = res.usedIndex;
      this.estimators[i] = new Estimator(this.treeOptions);
      this.estimators[i].train(X, y);

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

  /**
   * Train a random forest with the given training set and labels in a distributed manner.
   * Training is separated into "slices", each of which handles training one or more trees.
   * Distributed computing is handled using DCP. A valid DCP key must be configured on the
   * machine running this code for distributed training to work.
   * @param {Object} dcpArgs
   * @param {Number} [dcpArgs.estimatorsPerSlice=5] - How many estimators to train in each slice. The final slice may have less than this.
   * @param {Object[]} [dcpArgs.computeGroups] - An array of DCP compute group info objects (each with a "joinKey" and "joinSecret" string). Jobs will be deployed to these groups. Otherwise will run on the public network.
   * @param {String} [dcpArgs.name] - Name of the DCP job to be deployed.
   * @param {String} [dcpArgs.description] - Description for the DCP job to be deployed.
   * @param {Matrix|Array} trainingSet
   * @param {Array} trainingValues
   */
  async distributedTrain(dcpArgs, trainingSet, trainingValues) {

  let {
    estimatorsPerSlice,
    computeGroups,
  } = dcpArgs;
  // TODO: validate DCP args (?)

  // DCP should only be initialized once. Ensures this is true even in async contexts.
  if (!dcpInitializer) {
    dcpInitializer = dcpClient.init()
  }
  const dcp = await dcpInitializer;

  // Prep general purpose vars
  const totalSlices = Math.ceil(this.nEstimators / estimatorsPerSlice);

  // Prep slice arguments
  const sliceSeeds = Utils.generateSeeds(this.seed, totalSlices);
  const sliceEstimators = Array(totalSlices).fill(estimatorsPerSlice);
  const leftoverEstimators = this.nEstimators % estimatorsPerSlice;
  if (leftoverEstimators) {
    sliceEstimators[sliceEstimators.length-1] = leftoverEstimators;
  }
  const sliceArgs = Array(totalSlices);
  for (let i = 0; i < totalSlices; ++i) {
    sliceArgs[i] = {
      seed: sliceSeeds[i],
      numEstimators: sliceEstimators[i],
    }
  }

  // Prep job-wide arguments
  trainingSet = Matrix.checkMatrix(trainingSet);
  [trainingSet, trainingValues] = Utils.validateRFTrainingInputs(this, trainingSet, trainingValues);
  trainingSet = trainingSet.to2DArray();
  const jobArgs = {
    isClassifier: this.isClassifier,
    maxFeatures: this.maxFeatures || trainingSet.columns,
    numberFeatures: trainingSet.columns,
    numberSamples: trainingSet.rows,
    useSampleBagging: this.useSampleBagging,
  }

  this.estimators = new Array(this.nEstimators);
  this.indexes = new Array(this.nEstimators);
  let oobResults = new Array(this.nEstimators);

  // DCP job setup
  const workParams = [jobArgs, trainingSet, trainingValues];
  const job = dcp.compute.for(sliceArgs, workFunction, workParams);
  job.requires('distributed-ml-random-forest'); // TODO: get this added to the package manager
  // TODO: check proper syntax for local modules ^^^
  if (computeGroups) {
    job.computeGroups = computeGroups; // TODO: ask if this is correct/necessary to do
  }
  job.public.name = dcpArgs.name || 'distributed-ml-random-forest';
  job.public.description = dcpArgs.description || `Training ${estimatorsPerSlice} trees for a random forest model`;
  // Log various DCP events for debugging purposes.
  job.on('result', (resultObj) => console.log(`Result ${resultObj.sliceNumber} recieved.`));
  job.on('console', console.log);
  job.on('accepted', () => console.log('Training job is accepted and running. Job ID: ', job.id));

  let jobResults;
  if (jobArgs.localExec) {
    jobResults = await job.localExec(1);
  } else {
    jobResults = await job.exec();
  }

  // TODO: collect, format, and assign trained estimators
  let estIdx = 0;
  for (let i = 0; i < jobResults.length; ++i) {
    const sliceResults = jobResults[i];
    for (let j = 0; j < sliceResults.jsonEstimators.length; ++j) {
      let newEstimator;
      if (this.isClassifier) {
        newEstimator = DTClassifier.load(sliceResults.jsonEstimators[j])
      } else {
        newEstimator = DTRegression.load(sliceResults.jsonEstimators[j])
      }

      this.estimators[estIdx] = newEstimator;
      estIdx++;
    }
  }

  // TODO: figure out where this needs to go
  if (!this.noOOB && this.useSampleBagging && oobResults.length > 0) {
    this.oobResults = Utils.collectOOB(
      oobResults,
      trainingValues,
      this.selection.bind(this),
    );
  }
}

  /**
   * Evaluate the feature importances for each tree in the ensemble
   * @return {Array} feature importances
   */
  featureImportance() {
    const trees = JSON.parse(JSON.stringify(this.estimators));
    const indexes = JSON.parse(JSON.stringify(this.indexes));
    let importance = [];

    function computeFeatureImportances(i, node) {
      // node.gain can be null or undefined
      if (!node || !('splitColumn' in node) || !(node.gain > 0)) return;
      let f = node.gain * node.numberSamples;
      if ('left' in node) {
        f -= (node.left.gain || 0) * (node.left.numberSamples || 0);
      }
      if ('right' in node) {
        f -= (node.right.gain || 0) * (node.right.numberSamples || 0);
      }
      importance[i][node.splitColumn] += f;
      if (node.left) {
        computeFeatureImportances(i, node.left);
      }
      if (node.right) {
        computeFeatureImportances(i, node.right);
      }
    }

    function normalizeImportances(i) {
      const s = importance[i].reduce((cum, v) => {
        return (cum += v);
      }, 0);
      importance[i] = importance[i].map((v) => {
        return v / s;
      });
    }

    for (let i = 0; i < trees.length; i++) {
      importance.push(new Array(this.numberFeatures).fill(0.0));
      computeFeatureImportances(i, trees[i].root);
      normalizeImportances(i);
    }

    let avgImportance = new Array(this.numberFeatures).fill(0.0);
    for (let i = 0; i < importance.length; i++) {
      for (let x = 0; x < this.numberFeatures; x++) {
        avgImportance[indexes[i][x]] += importance[i][x];
      }
    }

    const s = avgImportance.reduce((cum, v) => {
      return (cum += v);
    }, 0);
    return avgImportance.map((v) => {
      return v / s;
    });
  }

  /**
   * Method that returns the way the algorithm generates the predictions, for example, in classification
   * you can return the mode of all predictions retrieved by the trees, or in case of regression you can
   * use the mean or the median.
   * @abstract
   * @param {Array} values - predictions of the estimators.
   * @return {number} prediction.
   */
  // eslint-disable-next-line no-unused-vars
  selection(values) {
    throw new Error("Abstract method 'selection' not implemented!");
  }

  /**
   * Predicts the output given the matrix to predict.
   * @param {Matrix|Array} toPredict
   * @return {Array} predictions
   */
  predict(toPredict) {
    const predictionValues = this.predictionValues(toPredict);
    let predictions = new Array(predictionValues.rows);
    for (let i = 0; i < predictionValues.rows; ++i) {
      predictions[i] = this.selection(predictionValues.getRow(i));
    }

    return predictions;
  }

  /**
   * Predicts the output given the matrix to predict.
   * @param {Matrix|Array} toPredict
   * @return {MatrixTransposeView} predictions of estimators
   */
  predictionValues(toPredict) {
    let predictionValues = new Array(this.nEstimators);
    toPredict = Matrix.checkMatrix(toPredict);
    for (let i = 0; i < this.nEstimators; ++i) {
      let X = new MatrixColumnSelectionView(toPredict, this.indexes[i]);
      predictionValues[i] = this.estimators[i].predict(X);
    }
    return (predictionValues = new MatrixTransposeView(
      new WrapperMatrix2D(predictionValues),
    ));
  }

  /**
   * Returns the Out-Of-Bag predictions.
   * @return {Array} predictions
   */
  predictOOB() {
    if (!this.oobResults || this.oobResults.length === 0) {
      throw new Error(
        'No Out-Of-Bag results found. Did you forgot to train first?',
      );
    }
    return this.oobResults.map((v) => v.predicted);
  }

  /**
   * Export the current model to JSON.
   * @return {object} - Current model.
   */
  toJSON() {
    return {
      indexes: this.indexes,
      n: this.n,
      replacement: this.replacement,
      maxFeatures: this.maxFeatures,
      nEstimators: this.nEstimators,
      treeOptions: this.treeOptions,
      isClassifier: this.isClassifier,
      seed: this.seed,
      estimators: this.estimators.map((est) => est.toJSON()),
      useSampleBagging: this.useSampleBagging,
    };
  }
}

export async function workFunction(sliceInput, jobArgs, trainingSet, trainingValues) {
  // TODO: "require" can be used here just fien, but we gotta make ESLint Happy
  const ml = require('ml'); // TODO: what do we need to actually import here?

  // extract info from slice and job arguments
  let [currentSeed, nEstimators] = sliceInput;
  let {
    modelParams
  } = jobArgs;
  let EstimatorClass;
  if (jobArgs.isClassifier) {
    EstimatorClass = DTClassifier;
  } else {
    EstimatorClass = DTRegression;
  }

  const estimators = new Array(sliceInput.nEstimators);
  const indexes = new Array(sliceInput.nEstimators);
  const oobResults = new Array(sliceInput.nEstimators);

  for (let i = 0; i < nEstimators; ++i) {
    progress(i / nEstimators);
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

    res = Utils.featureBagging(X, modelParams.n, modelParams.replacement, currentSeed);
    X = res.X;
    currentSeed = res.seed;

    // TODO: what are these supposed to do?
    indexes[i] = res.usedIndex;
    estimators[i] = new EstimatorClass(modelParams.treeOptions);
    estimators[i].train(X, y);

    // TODO: fix OOB issue (see issues section of original repo)
    if (!modelParams.noOOB && modelParams.useSampleBagging) {
      let xoob = new MatrixColumnSelectionView(Xoob, indexes[i]);
      oobResults[i] = {
        index: ioob,
        predicted: estimators[i].predict(xoob),
      };
    }
  }

  // TODO: need to confirm this works
  const jsonEstimators = estimators.map(estimator => JSON.stringify(estimator));

  const sliceResult = {
    indexes,
    jsonEstimators,
    oobResults,
  }

  return sliceResult;
}