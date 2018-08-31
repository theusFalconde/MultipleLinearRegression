package io.trackmotion;

import java.util.LinkedHashMap;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;

public class Fit {
	public static Model gradientDescent(Observation[] inputVectors, String dependent, double alpha) {
		Standardisation stanData = standardise(inputVectors);

		double[][] train = new double[inputVectors[0].size() + 1][inputVectors.length];
		for (int i = 0; i < inputVectors.length; i++) {
			train[0][i] = 1; // Intercept
			train[train.length - 1][i] = inputVectors[i].getFeature(dependent); // Dependent variable in the last column
			int j = 1;
			for (String feature : inputVectors[i].getFeatures()) {
				if (!feature.equals(dependent)) {
					train[j][i] = inputVectors[i].getFeature(feature);
					j++;
				}
			}
		}

		double[] thetas = new double[inputVectors[0].size()];
		double[] temps = new double[thetas.length];
		double delta;

		do {
			delta = 0;
			for (int i = 0; i < thetas.length; i++) {
				temps[i] = thetas[i] - (alpha * ((double) 1 / train.length) * evaluateCost(thetas, train, i));

				delta += Math.abs(thetas[i] - temps[i]);
			}

			for (int i = 0; i < thetas.length; i++)
				thetas[i] = temps[i];
		} while (delta > 1E-7);

		deStandardise(stanData, inputVectors, thetas);

		LinkedHashMap<String, Double> parameters = new LinkedHashMap<String, Double>();
		parameters.put("Intercept", thetas[0]);
		int j = 1;
		for (String feature : inputVectors[1].getFeatures()) {
			if (!feature.equals(dependent)) {
				parameters.put(feature, thetas[j]);
				j++;
			}
		}

		Model outputModel = new Model(parameters, dependent, 0);

		outputModel.rSquared = calculateRSquared(inputVectors, outputModel);

		return outputModel;
	}

	private static double evaluateCost(double[] thetas, double[][] data, int featureIndex) {
		double result = 0;

		for (int i = 0; i < data[0].length; i++) {
			double error = 0;

			for (int j = 0; j < data.length - 1; j++)
				error += data[j][i] * thetas[j];

			error -= data[data.length - 1][i];
			error *= data[featureIndex][i];

			result += error;
		}

		return result;
	}

	public static Model normalEquation(Observation[] inputVectors, String dependent) {
		double[][] design = new double[inputVectors.length][inputVectors[0].size()];
		double[][] designT = new double[inputVectors[0].size()][inputVectors.length];
		for (int i = 0; i < inputVectors.length; i++) {
			design[i][0] = 1;
			designT[0][i] = 1;
			int j = 1;
			for (String feature : inputVectors[i].getFeatures()) {
				if (!feature.equals(dependent)) {
					design[i][j] = inputVectors[i].getFeature(feature);
					designT[j][i] = inputVectors[i].getFeature(feature);
					j++;
				}
			}
		}
		RealMatrix X = new Array2DRowRealMatrix(design);
		RealMatrix XPrime = new Array2DRowRealMatrix(designT);

		double[] yArray = new double[inputVectors.length];
		for (int i = 0; i < inputVectors.length; i++)
			yArray[i] = inputVectors[i].getFeature(dependent);
		RealMatrix y = new Array2DRowRealMatrix(yArray);

		RealMatrix theta = new LUDecomposition(XPrime.multiply(X)).getSolver().getInverse().multiply(XPrime)
				.multiply(y);

		LinkedHashMap<String, Double> parameters = new LinkedHashMap<String, Double>();
		double[] thetas = theta.getColumn(0);
		parameters.put("Intercept", thetas[0]);
		int j = 1;
		for (String feature : inputVectors[1].getFeatures()) {
			if (!feature.equals(dependent)) {
				parameters.put(feature, thetas[j]);
				j++;
			}
		}

		Model outputModel = new Model(parameters, dependent, 0);

		outputModel.rSquared = calculateRSquared(inputVectors, outputModel);

		return outputModel;
	}

	private static class Standardisation {
		public Observation[] observations;
		public double[] xbars;
		public double[] sigmas;

		public Standardisation(Observation[] observations, double[] xbars, double[] sigmas) {
			this.observations = observations;
			this.xbars = xbars;
			this.sigmas = sigmas;
		}
	}

	private static Standardisation standardise(Observation[] inputVectors) {
		double[] xbars = new double[inputVectors[0].size()];
		double[] sigmas = new double[inputVectors[0].size()];

		for (int i = 0; i < inputVectors.length; i++) {
			int j = 0;
			for (String feature : inputVectors[i].getFeatures()) {
				xbars[j] += inputVectors[i].getFeature(feature);
				j++;
			}
		}

		for (int i = 0; i < xbars.length; i++)
			xbars[i] = xbars[i] / inputVectors.length;

		for (int i = 0; i < inputVectors.length; i++) {
			int j = 0;
			for (String feature : inputVectors[i].getFeatures()) {
				sigmas[j] += Math.pow(inputVectors[i].getFeature(feature) - xbars[j], 2);
				j++;
			}
		}

		for (int i = 0; i < sigmas.length; i++)
			sigmas[i] = Math.sqrt(sigmas[i] / inputVectors.length);

		for (int i = 0; i < inputVectors.length; i++) {
			int j = 0;
			for (String feature : inputVectors[i].getFeatures()) {
				inputVectors[i].putFeature(feature, (inputVectors[i].getFeature(feature) - xbars[j]) / sigmas[j]);
				j++;
			}
		}

		Standardisation output = new Standardisation(inputVectors, xbars, sigmas);

		return output;
	}

	private static void deStandardise(Standardisation standard, Observation[] inputVectors, double[] thetas) {
		for (int i = 1; i < thetas.length; i++) {
			thetas[0] -= thetas[i] * (standard.xbars[i - 1] / standard.sigmas[i - 1]);
			thetas[i] = (thetas[i] * standard.sigmas[standard.sigmas.length - 1]) / standard.sigmas[i - 1];
		}
		thetas[0] *= standard.sigmas[standard.sigmas.length - 1];
		thetas[0] += standard.xbars[standard.xbars.length - 1];

		for (int i = 0; i < inputVectors.length; i++) {
			int j = 0;
			for (String feature : inputVectors[i].getFeatures()) {
				inputVectors[i].putFeature(feature,
						((inputVectors[i].getFeature(feature) * standard.sigmas[j]) + standard.xbars[j]));
				j++;
			}
		}
	}

	private static double calculateRSquared(Observation[] inputVectors, Model model) {
		double ybar = 0;
		for (int i = 0; i < inputVectors.length; i++) {
			ybar += inputVectors[i].getFeature(model.dependent);
		}
		ybar /= inputVectors.length;

		double rss = 0;
		double tss = 0;
		for (int i = 0; i < inputVectors.length; i++) {
			rss += Math.pow((inputVectors[i].getFeature(model.dependent) - model.predict(inputVectors[i])), 2);
			tss += Math.pow((inputVectors[i].getFeature(model.dependent) - ybar), 2);
		}

		return (1 - rss / tss);
	}
}
