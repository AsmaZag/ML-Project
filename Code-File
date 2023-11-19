from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import yagmail

spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()


# Data Import and text processing:

new_dataset = spark.read.csv("chat_dataset.csv", header=True)
new_dataset = new_dataset.na.drop() 
new_dataset.show(10)

# Features preparations
tokenize = Tokenizer(inputCol="message", outputCol="words")
tokenized = tokenize.transform(new_dataset)

#Suppression des mots vides 
stopRemove = StopWordsRemover(inputCol="words", outputCol="filtered")
stopRemoved = stopRemove.transform(tokenized)



#Vectorisation 
vectorize = CountVectorizer(inputCol="filtered", outputCol="countVec")
counted_vec = vectorize.fit(stopRemoved).transform(stopRemoved)



#Inverse Document Frequency
idf = IDF(inputCol="countVec", outputCol="features")
rescale = idf.fit(counted_vec).transform(counted_vec)



# Indexation de la colonne d'étiquettes
indexer = StringIndexer(inputCol="sentiment", outputCol="Index")
indexed = indexer.fit(rescale).transform(rescale)
indexed.show()



# Train and test split
(training, testing) = indexed.randomSplit([0.8, 0.2], seed=42)

# Fit the model
LoReg = LogisticRegression(featuresCol="features", labelCol="Index")
LoRegModel = LoReg.fit(training)

# Test the model
test_pred = LoRegModel.transform(testing)
test_pred.select("features", "sentiment", "Index", "prediction").show(100)



# Evaluate accuracy
predictions = LoRegModel.transform(testing)
evaluator = MulticlassClassificationEvaluator(labelCol="Index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Display accuracy
print("Accuracy: {:.2%}".format(accuracy))


# Visualization
sentiment_counts = predictions.groupBy("prediction").count().toPandas()
plt.figure(figsize=(8, 5))
plt.bar(sentiment_counts["prediction"], sentiment_counts["count"], color=['red', 'green', 'blue'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(sentiment_counts["prediction"], ["Negative", "Neutral", "Positive"])
plt.show()

# Trigger and notification

threshold = 0.8
if accuracy < threshold:
    # Send an email alert
    sender_email = "asmazagmouzi@gmail.com"
    receiver_email = "asmazagmouzi@gmail.com"
    app_password = "€€€€€€"

    subject = "Sentiment Analysis Alert"
    body = f"The sentiment accuracy has fallen below the threshold. Current accuracy: {accuracy:.2%}"

    try:
        # Connect to yagmail SMTP server
        yag = yagmail.SMTP(sender_email, app_password)

        # Send email
        yag.send(to=receiver_email, subject=subject, contents=body)

        print("Alert sent!")
    except Exception as e:
        print(f"Error sending alert: {e}")
else:
    print("No alert triggered. Current accuracy:", accuracy)
