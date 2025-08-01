import { Matrix, MatrixColumnSelectionView } from 'ml-matrix';

import * as Utils from '../utils';

let rows = 40;
let cols = 25;
let testX = Matrix.rand(rows, cols);
let testY = new Array(rows).fill(1);

describe('Utils', () => {
  it('Retrieve features', () => {
    let data = new Matrix([
      [1, 2, 3, 4, 5],
      [1, 2, 3, 4, 5],
      [1, 2, 3, 4, 5],
    ]);
    let indexes = [0, 4];

    let newData = new MatrixColumnSelectionView(new Matrix(data), indexes);
    for (let i = 0; i < newData.column; ++i) {
      expect(newData[i]).toBe([1, 5]);
    }
  });

  it('Examples bagging', () => {
    let data = Utils.examplesBaggingWithReplacement(testX, testY);

    expect(data.X).toBeInstanceOf(Matrix);
  });

  it('Feature bagging with replacement', () => {
    let data = Utils.featureBagging(testX, cols - 5, true, 7);
    expect(new Set(data.usedIndex).size).toBeLessThan(data.usedIndex.length);
    expect(data.X.columns).toBe(20);
  });

  it('Feature bagging without replacement', () => {
    let data = Utils.featureBagging(testX, cols - 5, false, 7);
    expect(new Set(data.usedIndex).size).toBe(cols - 5);
    expect(data.X.columns).toBe(20);
  });

  describe('generateSeeds()', () => {

    it('should generate n numerical seeds', () => {
      const seeds = Utils.generateSeeds(400, 10);
      expect(seeds).toHaveLength(10);
      for (let seed of seeds) {
        expect(typeof seed).toBe('number');
        expect(seed).not.toBeNaN();
        expect(Math.floor(seed)).toStrictEqual(seed); // is integer
      }
    });

    it('should generate unique seeds', () => {
      const seeds = Utils.generateSeeds(500, 10);
      const seenSeeds = {};
      for (let seed of seeds) {
        expect(seenSeeds[seed]).toBeUndefined();
        seenSeeds[seed] = true;
      }
    });

    it('should generate different seeds for a different input seed', () => {
      const seeds1 = Utils.generateSeeds(500, 10);
      const seeds2 = Utils.generateSeeds(600, 10);
      for (let i in seeds1) {
        expect(seeds1[i]).not.toStrictEqual(seeds2[i]);
      }
    });

    it('should generate the same seeds sequence for a different input size with the same seed', () => {
      const seeds1 = Utils.generateSeeds(700, 5);
      const seeds2 = Utils.generateSeeds(700, 10);
      for (let i in seeds1) {
        expect(seeds1[i]).toStrictEqual(seeds2[i]);
      }
    });

    it('should generate seeds reproducibly', () => {
      const seeds1 = Utils.generateSeeds(1000, 10);
      const seeds2 = Utils.generateSeeds(1000, 10);
      expect(seeds1).toStrictEqual(seeds2);
    });
  });
});
