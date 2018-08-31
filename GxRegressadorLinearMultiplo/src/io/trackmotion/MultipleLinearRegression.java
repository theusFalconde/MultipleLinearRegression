package io.trackmotion;

// FIXME Para executar o código primeiro deve-se apagar as colunas de data e hora, e os valores que contenham , devem ser substituidos por .
// FIXME Na pasta do projeto existe o diretório DATA, onde é possível encontrar o arquivo AirQualityUCI, já com as alterações citadas acima
public class MultipleLinearRegression {
	public static void main(String[] args) {
		Observation[] inputVectors = FileHandler.read("data\\AirQualityUCI.csv");

		double alpha = 0.0001;
		
		//FIXME Indique na variável abaixo a coluna do CSV que deseja realizar a predição. Pronto Agora só esperar. Pode demorar alguns minutos para que a previsão termine, tenha paciencia.
		String feature = "NO2(GT)";
		
		System.out.println("\n\nModelo usando gradiente descendente, alpha = " + alpha + " ...");
		System.out.println("*************************************************************************\n");
		Model gradientFit = Fit.gradientDescent(inputVectors, feature, 0.0001);
		System.out.println(gradientFit.toString());

		System.out.println("\n\nModelo usando o método de equação normal");
		System.out.println("*************************************************************************\n");
		Model normalFit = Fit.normalEquation(inputVectors, feature);
		System.out.println(normalFit.toString());

		System.out.println("\n\nPrevendo algumas linhas arbitrárias usando os dois modelos...\n");
		double testValue = inputVectors[30].getFeature(feature);
		double predictionA = gradientFit.predict(inputVectors[30]);
		double predictionB = normalFit.predict(inputVectors[30]);
		System.out.println("Valor atual: " + testValue + "\nPrevisão usando gradiente descendente: " + predictionA);
		System.out.println("Previsão usando o método de equação normal: " + predictionB);
		testValue = inputVectors[40].getFeature(feature);
		predictionA = gradientFit.predict(inputVectors[40]);
		predictionB = normalFit.predict(inputVectors[40]);
		System.out.println("Valor atual: " + testValue + "\nPrevisão usando gradiente descendente: " + predictionA);
		System.out.println("Previsão usando o método de equação normal: " + predictionB);
		testValue = inputVectors[50].getFeature(feature);
		predictionA = gradientFit.predict(inputVectors[50]);
		predictionB = normalFit.predict(inputVectors[50]);
		System.out.println("Valor atual: " + testValue + "\nPrevisão usando gradiente descendente: " + predictionA);
		System.out.println("Previsão usando o método de equação normal: " + predictionB);
	}
}
