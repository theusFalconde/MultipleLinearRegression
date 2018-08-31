package io.trackmotion;

public class MultipleLinearRegression {
	public static void main(String[] args) {
		Observation[] inputVectors = FileHandler.read("C:\\Users\\mathe\\Downloads\\AirQualityUCI\\AirQualityUCI.csv");

		double alpha = 0.0001;
		System.out.println("\n\nFitting model using gradient descent, alpha = " + alpha + " ...");
		System.out.println("*************************************************************************\n");
		Model gradientFit = Fit.gradientDescent(inputVectors, "NO2(GT)", 0.0001);
		System.out.println(gradientFit.toString());

		System.out.println("\n\nFitting model in using the normal equation method ...");
		System.out.println("*************************************************************************\n");
		Model normalFit = Fit.normalEquation(inputVectors, "NO2(GT)");
		System.out.println(normalFit.toString());

		System.out.println("\n\nPredicting some arbitrary rows using both models ...\n");
		double testValue = inputVectors[30].getFeature("NO2(GT)");
		double predictionA = gradientFit.predict(inputVectors[30]);
		double predictionB = normalFit.predict(inputVectors[30]);
		System.out.println("Actual value: " + testValue + "\nPrediction using gradient descent: " + predictionA);
		System.out.println("Prediction using normal equation method: " + predictionB);
		testValue = inputVectors[40].getFeature("NO2(GT)");
		predictionA = gradientFit.predict(inputVectors[40]);
		predictionB = normalFit.predict(inputVectors[40]);
		System.out.println("Actual value: " + testValue + "\nPrediction using gradient descent: " + predictionA);
		System.out.println("Prediction using normal equation method: " + predictionB);
		testValue = inputVectors[50].getFeature("NO2(GT)");
		predictionA = gradientFit.predict(inputVectors[50]);
		predictionB = normalFit.predict(inputVectors[50]);
		System.out.println("Actual value: " + testValue + "\nPrediction using gradient descent: " + predictionA);
		System.out.println("Prediction using normal equation method: " + predictionB);
	}
}
