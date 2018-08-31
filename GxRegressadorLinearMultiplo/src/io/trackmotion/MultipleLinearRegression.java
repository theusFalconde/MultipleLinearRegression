package io.trackmotion;

// FIXME Para executar o c�digo primeiro deve-se apagar as colunas de data e hora, e os valores que contenham , devem ser substituidos por .
// FIXME Na pasta do projeto existe o diret�rio DATA, onde � poss�vel encontrar o arquivo AirQualityUCI, j� com as altera��es citadas acima
public class MultipleLinearRegression {
	public static void main(String[] args) {
		Observation[] inputVectors = FileHandler.read("data\\AirQualityUCI.csv");

		double alpha = 0.0001;
		
		//FIXME Indique na vari�vel abaixo a coluna do CSV que deseja realizar a predi��o. Pronto Agora s� esperar. Pode demorar alguns minutos para que a previs�o termine, tenha paciencia.
		String feature = "NO2(GT)";
		
		System.out.println("\n\nModelo usando gradiente descendente, alpha = " + alpha + " ...");
		System.out.println("*************************************************************************\n");
		Model gradientFit = Fit.gradientDescent(inputVectors, feature, 0.0001);
		System.out.println(gradientFit.toString());

		System.out.println("\n\nModelo usando o m�todo de equa��o normal");
		System.out.println("*************************************************************************\n");
		Model normalFit = Fit.normalEquation(inputVectors, feature);
		System.out.println(normalFit.toString());

		System.out.println("\n\nPrevendo algumas linhas arbitr�rias usando os dois modelos...\n");
		double testValue = inputVectors[30].getFeature(feature);
		double predictionA = gradientFit.predict(inputVectors[30]);
		double predictionB = normalFit.predict(inputVectors[30]);
		System.out.println("Valor atual: " + testValue + "\nPrevis�o usando gradiente descendente: " + predictionA);
		System.out.println("Previs�o usando o m�todo de equa��o normal: " + predictionB);
		testValue = inputVectors[40].getFeature(feature);
		predictionA = gradientFit.predict(inputVectors[40]);
		predictionB = normalFit.predict(inputVectors[40]);
		System.out.println("Valor atual: " + testValue + "\nPrevis�o usando gradiente descendente: " + predictionA);
		System.out.println("Previs�o usando o m�todo de equa��o normal: " + predictionB);
		testValue = inputVectors[50].getFeature(feature);
		predictionA = gradientFit.predict(inputVectors[50]);
		predictionB = normalFit.predict(inputVectors[50]);
		System.out.println("Valor atual: " + testValue + "\nPrevis�o usando gradiente descendente: " + predictionA);
		System.out.println("Previs�o usando o m�todo de equa��o normal: " + predictionB);
	}
}
