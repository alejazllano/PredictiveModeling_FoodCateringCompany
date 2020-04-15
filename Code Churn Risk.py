# Databricks notebook source
# DBTITLE 1,Predictive Churn Model for BoBo Company
# Members:
# Alejandro Lopez
# Chenxin Xie
# Alejandra Zambrano

# MBD 2019_2020
# BIG DATA TOOLS 2

# COMMAND ----------

#Set path for reading the data
compliantsFilePath= "/FileStore/tables/BDT2_1920_Complaints.csv"
customersFilePath= "/FileStore/tables/BDT2_1920_Customers.csv"
formulaFilePath= "/FileStore/tables/BDT2_1920_Formula.csv"
subscriptionsFilePath= "/FileStore/tables/BDT2_1920_Subscriptions.csv"
deliveryFilePath= "/FileStore/tables/BDT2_1920_Delivery.csv"

# COMMAND ----------

#Importing modules and sub-modules
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# COMMAND ----------

# DBTITLE 1,Data Preparation and Base Table Creation
#Reading Complaints table
complaints=spark.read.format("csv").option("header","true").option("inferSchema","true").load(compliantsFilePath)
complaints.printSchema()

# COMMAND ----------

#Grouping each row by CustomerID, displaying the number of Complaints by Complaint Type, date of last complaint, total number of complaints, number of solutions by type and feedback

#Filter before a date
last_date="2018-01-30" #To create the train 
start_date="2014-01-30" #To create the train 


#Complaints by type
complaints_user=complaints.withColumn("ComplaintDate", to_date(col("ComplaintDate"), "yyyy-MM-dd")).filter(col("ComplaintDate").between(lit(start_date),lit(last_date)))\
  .groupBy("CustomerID").pivot("ComplaintTypeDesc").agg(count("ComplaintID"))

#Last complaint and total number of complaints
last_complaint=complaints.withColumn("ComplaintDate", to_date(col("ComplaintDate"), "yyyy-MM-dd")).filter(col("ComplaintDate").between(lit(start_date),lit(last_date)))\
  .groupBy("CustomerID").agg(expr("count(ComplaintID) as total_complaints"),expr("max(ComplaintDate) as last_complaint"))

complaints_user=complaints_user.join(last_complaint,"CustomerID","left_outer").withColumnRenamed("billing/invoice not correct","nbrComp_billing_incorrect").withColumnRenamed("employees were rude","nbrComp_rude_employees")\
  .withColumnRenamed("food quality not good","nbrComp_food_quality").withColumnRenamed("food quantity was insufficient","nbrComp_food_quantity").withColumnRenamed("food was cold","nbrComp_food_cold")\
  .withColumnRenamed("late delivery","nbrComp_late_delivery").withColumnRenamed("order not correct","nbrComp_incorrect_order").withColumnRenamed("poor hygiene","nbrComp_hygiene_food")\
  .withColumnRenamed("other","nbrComp_other_complaint")

#Complaints' solution
complaints_solution=complaints.withColumn("ComplaintDate", to_date(col("ComplaintDate"), "yyyy-MM-dd")).filter(col("ComplaintDate").between(lit(start_date),lit(last_date)))\
  .groupBy("CustomerID").pivot("SolutionTypeDesc").agg(count("ComplaintID"))

complaints_user=complaints_user.join(complaints_solution,"CustomerID","left_outer").withColumnRenamed("NA","nbrSln_na_solution").withColumnRenamed("exceptional price discount","nbrSln_price_discount")\
  .withColumnRenamed("free additional meal service","nbrSln_aditional_meal").withColumnRenamed("no compensation","nbrSln_no_compensation").withColumnRenamed("other","nbrSln_other_solution")

#Complaints' Feedback
complaints_feedback=complaints.withColumn("ComplaintDate", to_date(col("ComplaintDate"), "yyyy-MM-dd")).filter(col("ComplaintDate").between(lit(start_date),lit(last_date)))\
  .groupBy("CustomerID").pivot("FeedbackTypeDesc").agg(count("ComplaintID"))

#Joining final complaint table, creating rate solution variable= complaints with solutions/total complaints and rate no satisfaction = no satisfied complaints/total complaints
complaints_user=complaints_user.join(complaints_feedback,"CustomerID","left_outer").withColumnRenamed("NA","na_feedback").withColumnRenamed("no response","no_ans_feedback")\
  .withColumnRenamed("not satisfied","no_satisfied").withColumnRenamed("other","other_feedback").withColumn("rate_sln", round((col("total_complaints")-col("nbrSln_na_solution"))/col("total_complaints"),3))\
  .withColumn("rate_no_satf", round((col("no_satisfied")/col("total_complaints")),3)).na.fill(0)


# COMMAND ----------

#Reading Customers table
customers=spark.read.format("csv").option("header","true").option("inferSchema","true").load(customersFilePath)
customers.printSchema()

#Counting number of disctint Customer ID - Customers
customers.select("CustomerID").distinct().count()

# COMMAND ----------

#Reading Formula table
formula = spark.read.format("csv").option("header","true").option("inferSchema","true").load(formulaFilePath)
formula.printSchema()

# COMMAND ----------

#Reading Subscription table
subscriptions=spark.read.format("csv").option("header","true").option("inferSchema","true").load(subscriptionsFilePath)
subscriptions.printSchema()

# COMMAND ----------

#Dropping NA values in columns GrossFormulaPrice - TotalCredit
subscriptions = subscriptions.filter((subscriptions.GrossFormulaPrice != 'NA'))

#Flag renewals
subscriptions=subscriptions.withColumn("renewals",when(col("RenewalDate")=='NA',0).otherwise(1))

#Flag TotalCredit
subscriptions=subscriptions.withColumn("credits",when(col("TotalCredit")=='0',0).otherwise(1))

#Joining subscriptions and Formula tables to have the duration of the formula in months
subscriptions=subscriptions.join(formula,"FormulaID","left_outer")

#Changing names products
subscriptions=subscriptions.withColumn("ProductName",when(col("ProductName")=="Custom Events","prod_custom_events").when(col("ProductName")=="Grub Flexi (excl. staff)","prod_flexi")\
  .when(col("ProductName")=="Grub Maxi (incl. staff)","prod_maxi").otherwise("prod_mini"))

#Last start_date before filter date and recency variable
maxStart=subscriptions.withColumn("StartDate", to_date(col("StartDate"), "yyyy-MM-dd")).filter(col("StartDate").between(lit(start_date),lit(last_date)))\
  .groupBy("CustomerID").agg(expr("max(StartDate) as max_SartDate"))
  
#Getting information last subscription
w = Window().partitionBy("CustomerID").orderBy(subscriptions.StartDate.desc())
subscriptions_last=subscriptions.withColumn("EndDateF", to_date(col("EndDate"), "yyyy-MM-dd")).filter(col("EndDateF").between(lit(start_date),lit(last_date)))\
  .select(col("CustomerID"), first("RenewalDate",True).over(w).alias('RenewalDate_last'),\
  first("StartDate",True).over(w).alias('StartDate_last'),\
  first("ProductName",True).over(w).alias('ProductName_last'),first("FormulaType",True).over(w).alias('FormulaType_last'),first("Duration",True).over(w).alias('duration_last'),\
  first("PaymentStatus",True).over(w).alias('PaymentStatus_last')).withColumn("StartDate_last", to_date(col("StartDate_last"), "yyyy-MM-dd")).distinct()

#Aggregate variables by customer
subscriptions_agg=subscriptions.withColumn("EndDateF", to_date(col("EndDate"), "yyyy-MM-dd")).filter(col("EndDateF").between(lit(start_date),lit(last_date)))\
  .groupBy("CustomerID").agg(expr("count(SubscriptionID) as total_subscriptions"),\
   expr("max(EndDateF) as date_last_Endsubs"),expr("avg(Duration) as avg_duration_sub"),expr("sum(renewals) as total_renewals"),\
   expr("sum(credits) as total_credits"),expr("avg(NbrMeals_REG) as avg_mealsREG"), expr("avg(NbrMeals_EXCEP) as avg_mealsEXCEP"),expr("avg(GrossFormulaPrice) as avg_form_price"),\
   expr("avg(NbrMealsPrice) as avg_price_meal"),expr("avg(ProductDiscount) as avg_product_dcto"),expr("avg(FormulaDiscount) as avg_formula_dcto"),expr("avg(TotalPrice) as avg_total_price"))

#Grouping each row by CustomerID and Product
subscriptions_product=subscriptions.withColumn("EndDateF", to_date(col("EndDate"), "yyyy-MM-dd")).filter(col("EndDateF").between(lit(start_date),lit(last_date)))\
  .groupBy("CustomerID").pivot("ProductName").agg(count("SubscriptionID"))

#Joining tables and creating recency variable
subscriptions_base=subscriptions_agg.join(subscriptions_last,"CustomerID","left_outer").join(maxStart,"CustomerID","left_outer").join(subscriptions_product,"CustomerID","left_outer")
subscriptions_base=subscriptions_base.withColumn("recency_endSubs",datediff(lit(last_date),col("date_last_Endsubs"))).withColumnRenamed("prod_custom_events","nbrProd_custom_events")\
  .withColumnRenamed("prod_flexi","nbrProd_flexi").withColumnRenamed("prod_maxi","nbrProd_maxi").withColumnRenamed("prod_mini","nbrProd_mini")


# COMMAND ----------

#Reading Delivery table
delivery = spark.read.format("csv").option("header","true").option("inferSchema","true").load(deliveryFilePath)
delivery.printSchema()

# COMMAND ----------

#Joining delivery and Subscription table
delivery_suscriptions=delivery.join(subscriptions,"SubscriptionID","left_outer")

# COMMAND ----------

#Grouping each row by SubscriptionID, displaying the number of total deliveries, deliveries by class, last delivery date and average days between deliveries

#Delivery by Class
delivery_base=delivery_suscriptions.withColumn("DeliveryDate", to_date(col("DeliveryDate"), "yyyy-MM-dd")).filter(col("DeliveryDate").between(lit(start_date),lit(last_date)))\
  .groupBy("CustomerID").pivot("DeliveryClass").agg(count("DeliveryID")).withColumnRenamed("null","missing_delivery_class").withColumnRenamed("ABN","nbr_abnormal_deliveries")\
  .withColumnRenamed("NOR","nbr_normal_deliveries")

#Last delivery and total number of deliveries
last_delivery=delivery_suscriptions.withColumn("DeliveryDate", to_date(col("DeliveryDate"), "yyyy-MM-dd")).filter(col("DeliveryDate").between(lit(start_date),lit(last_date)))\
   .groupBy("CustomerID").agg(expr("count(DeliveryID) as total_deliveries"),expr("max(DeliveryDate) as last_delivery"))

delivery_base=delivery_base.join(last_delivery,"CustomerID", "left_outer")

#Avg days between deliveries by CustomerID
delivery_days_subs=delivery_suscriptions.withColumn("DeliveryDateF", to_date(col("DeliveryDate"), "yyyy-MM-dd")).filter(col("DeliveryDateF").between(lit(start_date),lit(last_date)))\
   .withColumn("days_int",((delivery_suscriptions.DeliveryDate.cast("bigint") - lag(delivery_suscriptions.DeliveryDate.cast("bigint"), 1).over(Window.partitionBy("CustomerID")\
   .orderBy("DeliveryDate")))/(24*3600)).cast("bigint")).groupBy("CustomerID").agg(expr("avg(days_int) as avg_days_inter_del"))

delivery_base=delivery_base.join(delivery_days_subs,"CustomerID", "left_outer").na.fill(0)

#Avg nbr deliveries by subscription in each customer
deliveries_sub=delivery_suscriptions.groupBy("CustomerID","SubscriptionID").agg(count("DeliveryID")).groupBy("CustomerID").agg(avg("count(DeliveryID)"))\
  .withColumnRenamed("avg(count(DeliveryID))","avg_nbr_deliveries_subs")

delivery_base=delivery_base.join(deliveries_sub,"CustomerID", "left_outer")


# COMMAND ----------

#Final Base Table
base_table=customers.join(subscriptions_base,"CustomerID","left_outer").join(delivery_base,"CustomerID","left_outer").join(complaints_user,"CustomerID","left_outer")

# COMMAND ----------

# DBTITLE 1,Create Labels, Dummy Variables and Column Features
#Dropping customers with Null values, because they are customers that start their suscription after 2018-01-31
base_table = base_table.filter(base_table.total_subscriptions. isNotNull())

# COMMAND ----------

#Label the base table with 1 for churn customers and 0 for non-churn customers
base_table=base_table.withColumn("label", when((col("RenewalDate_last")!='NA')|(col("max_SartDate")> col("date_last_Endsubs")),0).otherwise(1))

# COMMAND ----------

base_table.printSchema()

# COMMAND ----------

#Fill null values with O
base_table=base_table.na.fill(0)

# COMMAND ----------

#Getting columns names
variables=base_table.columns

#Items to be removed
remove = ['CustomerID','Region','RenewalDate_last','StartDate_last','ProductName_last','FormulaType_last','PaymentStatus_last','max_SartDate','last_delivery','last_complaint','date_last_Endsubs','label', 'StreetID']
variables = [ele for ele in variables if ele not in remove] 

# COMMAND ----------

#Creating dummy variables and feature column
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

ProductName_lastIndxr = StringIndexer().setHandleInvalid("skip").setInputCol("ProductName_last").setOutputCol("ProductName_last_idx")
FormulaType_lastIndxr = StringIndexer().setInputCol("FormulaType_last").setOutputCol("FormulaType_last_idx")
RegionIndxr = StringIndexer().setInputCol("Region").setOutputCol("Region_idx")
onehot = OneHotEncoderEstimator(inputCols=['ProductName_last_idx','FormulaType_last_idx','Region_idx'],\
         outputCols=['ProductName_last_dummy','FormulaType_last_dummy','Region_dummy'])

variables2=['ProductName_last_dummy','FormulaType_last_dummy','Region_dummy']
variables=variables+variables2

assemble = VectorAssembler().setInputCols(variables).setOutputCol("all_features")

pipe_catv = Pipeline(stages=[ProductName_lastIndxr,FormulaType_lastIndxr,RegionIndxr,onehot,assemble])

base_trans=pipe_catv.fit(base_table).transform(base_table)

# COMMAND ----------

base_trans=base_trans.drop('ProductName_last_idx','FormulaType_last_idx','Region_idx','date_last_Endsubs','StreetID','RenewalDate_last','StartDate_last','ProductName_last','FormulaType_last',\
  'PaymentStatus_last','Region','max_SartDate','last_complaint')

# COMMAND ----------

# DBTITLE 1,Create Train and Test Set
#Split base table in stratifieed sample for train and test set 

train = base_trans.sampleBy("label", fractions={0: 0.8, 1: 0.8}, seed=123)
# Subtracting 'train' from original 'data' to get test set
test = base_trans.subtract(train)
# Checking distributions of 0's and 1's in train and test sets after the sampling
train.groupBy("label").count().show()
test.groupBy("label").count().show()

# COMMAND ----------

# DBTITLE 1,Feature Selection
#Performing Chi Square Selector
from pyspark.ml.feature import ChiSqSelector

chisq = ChiSqSelector().setFeaturesCol("all_features").setLabelCol("label").setOutputCol("select_features").setNumTopFeatures(15)
train_feature=chisq.fit(train).transform(train)

# COMMAND ----------

#Top features selected using Chi Square 
model = chisq.fit(train)
importantFeatures = model.selectedFeatures

def gname(variables):
  select_features = []
  for i in importantFeatures:
    select_features.append(variables[i])
  return select_features

# COMMAND ----------

importantFeatures

# COMMAND ----------

select_features=gname(variables)

# COMMAND ----------

# DBTITLE 1,Standarized Variables
#Standarize train set (mean=0, sd=1)
from pyspark.ml.feature import StandardScaler

sScaler = StandardScaler().setWithStd(True).setWithMean(True).setInputCol("select_features").setOutputCol("features")
train_stnd=sScaler.fit(train_feature).transform(train_feature)

# COMMAND ----------

#Assemble select features in the test set
assemble_select = VectorAssembler().setInputCols(select_features).setOutputCol("select_features")
test_select=assemble_select.transform(test)

# COMMAND ----------

#Standarize test set (mean=0, sd=1)
from pyspark.ml.feature import StandardScaler

sScaler = StandardScaler().setWithStd(True).setWithMean(True).setInputCol("select_features").setOutputCol("features")
test_stnd=sScaler.fit(test_select).transform(test_select)

# COMMAND ----------

# DBTITLE 1,Modeling: Random Forest Classifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

#Define pipeline
rfc = RandomForestClassifier()
rfPipe = Pipeline().setStages([rfc])

#Set param grid
rfParams = ParamGridBuilder()\
  .addGrid(rfc.numTrees, [150, 300, 500])\
  .build()

rfCv = CrossValidator()\
  .setEstimator(rfPipe)\
  .setEstimatorParamMaps(rfParams)\
  .setEvaluator(BinaryClassificationEvaluator())\
  .setNumFolds(2) 

#Run cross-validation, and choose the best set of parameters.
rfcModelSTD = rfCv.fit(train_stnd)

# COMMAND ----------

#Making predictions with the model Random Forest Classifier
preds = rfcModelSTD.transform(test_stnd).select("prediction", "label")

# COMMAND ----------

#Get model performance on test set
from pyspark.mllib.evaluation import BinaryClassificationMetrics

out = preds.rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = BinaryClassificationMetrics(out)

print(metrics.areaUnderPR) #area under precision/recall curve
print(metrics.areaUnderROC)#area under Receiver Operating Characteristic curve

# COMMAND ----------

# DBTITLE 1,Modeling: Logistic Regression 
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

#Define pipeline
lr = LogisticRegression().setMaxIter(5)
lrPipe = Pipeline().setStages([lr])

#Set param grid
lrParams = ParamGridBuilder()\
  .addGrid(lr.regParam, [0.1, 0.01])\
  .build()

#Evaluator to get the final model
evaluatorlr = BinaryClassificationEvaluator()

lrCv = CrossValidator()\
  .setEstimator(lrPipe)\
  .setEstimatorParamMaps(lrParams)\
  .setEvaluator(evaluatorlr)\
  .setNumFolds(2)

#Run cross-validation, and choose the best set of parameters
lrModel = lrCv.fit(train_stnd)

# COMMAND ----------

#Making predictions with the model Random Forest Classifier
preds = lrModel.transform(test_stnd).select("prediction", "label")

# COMMAND ----------

#Get model performance on test set
from pyspark.mllib.evaluation import BinaryClassificationMetrics

out = preds.rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = BinaryClassificationMetrics(out)

print(metrics.areaUnderPR) #area under precision/recall curve
print(metrics.areaUnderROC)#area under Receiver Operating Characteristic curve

# COMMAND ----------

# DBTITLE 1,Interpretability Selected Model
#Select the best RF model
rfcBestModel = rfcModelSTD.bestModel.stages[-1]
#Get tuned number of trees
rfcBestModel.getNumTrees

# COMMAND ----------

#Get feature importances
rfcBestModel.featureImportances

# COMMAND ----------

#Feature importances
import pandas as pd
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))
  
ExtractFeatureImp(rfcBestModel.featureImportances, train_feature, "select_features").head(15)

# COMMAND ----------

# DBTITLE 1,Applying Selected Model in Current Costumers
# We change the last date and start day for generating the base table and apply the select model

# last_date="2019-03-30"  
# start_date="2015-03-30"  

actual_clients=base_table

# COMMAND ----------

#Calculating the end date of the last renewed subscription
actual_clients=actual_clients.withColumn('duration_days',30*col("duration_last"))
actual_clients=actual_clients.withColumn("lastEnd_aprox", expr("date_add(date_last_Endsubs,duration_days)"))

# COMMAND ----------

#Label costumers (1 and 0) with an active susbcription to determine the probability to churn
actual_clients=actual_clients.withColumn("active_costumer",when((col("lastEnd_aprox")>=lit("2019-02-15")) & (col("RenewalDate_last")!='NA'),1).otherwise(0))

# COMMAND ----------

#Filter by active costumers (=1) and fill null values with zero
actual_clients=actual_clients.filter(col("active_costumer")==1)
actual_clients=actual_clients.na.fill(0)

# COMMAND ----------

#Assemble selected features in the actual clients for applying the selected model
assemble_select = VectorAssembler().setInputCols(select_features).setOutputCol("features_select")
actual_predict=assemble_select.transform(actual_clients)

# COMMAND ----------

#Standarize actual set (mean=0, sd=1)
from pyspark.ml.feature import StandardScaler

sScaler = StandardScaler().setWithStd(True).setWithMean(True).setInputCol("features_select").setOutputCol("features")
actual_predict=sScaler.fit(actual_predict).transform(actual_predict)

# COMMAND ----------

#Making predictions with the model Random Forest Classifier
predictions = rfcModelSTD.transform(actual_predict)
