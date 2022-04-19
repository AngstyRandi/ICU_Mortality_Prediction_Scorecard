pacman::p_load(ggplot2, dplyr, magrittr)


## Load Data
getwd()
setwd('/Users/Caroline/Desktop/EBAC/04-PredictiveModellingandForcasting/PracticeModule/R')
train.data = read.csv("training_v2.csv")
str(train.data)


## Check Missing Value
sapply(train.data, function(x) sum(is.na(x)))
sapply(train.data, function(x) sum(is.na(x))/nrow(train.data))
sapply(train.data, function(x) levels(x))


## Select Columns
columns = c("encounter_id","hospital_death","age","bmi","elective_surgery","ethnicity","gender","height","hospital_admit_source",
            "icu_admit_source","icu_id","icu_stay_type","icu_type","pre_icu_los_days","readmission_status","weight",
            "apache_4a_hospital_death_prob","apache_4a_icu_death_prob","aids","cirrhosis","diabetes_mellitus",
            "hepatic_failure","immunosuppression","leukemia","lymphoma","solid_tumor_with_metastasis",
            "apache_3j_bodysystem","apache_2_bodysystem")
fwl.train.data = train.data[columns]


## Data Cleaning
# Delete the whole row for age, gender is missing
# fwl.train.data = fwl.train.data %>%
#   filter(!is.na(age))
# fwl.train.data = fwl.train.data[!(is.na(fwl.train.data$gender)| fwl.train.data$gender==""),]

# Categorize BMI
summary(fwl.train.data$bmi)
fwl.train.data$bmi_name = NA
fwl.train.data$bmi_name[is.na(fwl.train.data$bmi)] = "unknown"
fwl.train.data$bmi_name[fwl.train.data$bmi <= 16] = "underweight"
fwl.train.data$bmi_name[fwl.train.data$bmi > 16 & fwl.train.data$bmi <= 18.5] = "normal"
fwl.train.data$bmi_name[fwl.train.data$bmi > 18.5] = "overweight"
fwl.train.data$bmi_name = factor(fwl.train.data$bmi_name, levels=c("unknown", "underweight", "normal", "overweight"))
str(fwl.train.data)
levels(fwl.train.data$bmi_name)

# Drop bmi,height,weight,readmission_status
columns = c("encounter_id","hospital_death","age","elective_surgery","ethnicity","gender","hospital_admit_source",
            "icu_admit_source","icu_id","icu_stay_type","icu_type","pre_icu_los_days",
            "apache_4a_hospital_death_prob","apache_4a_icu_death_prob","aids","cirrhosis","diabetes_mellitus",
            "hepatic_failure","immunosuppression","leukemia","lymphoma","solid_tumor_with_metastasis",
            "apache_3j_bodysystem","apache_2_bodysystem")
fwl.train.data = fwl.train.data[columns]


# Fill NA for categorical variable
fill_categorical_na = function(column_data, value) {
  column_data = as.character(column_data)
  column_data[column_data == ""] = value
  column_data = factor(column_data)
  return (column_data)
}

fwl.train.data$ethnicity = fill_categorical_na(fwl.train.data$ethnicity,"Other/Unknown")
fwl.train.data$hospital_admit_source = fill_categorical_na(fwl.train.data$hospital_admit_source,"Unknown")
fwl.train.data$icu_admit_source = fill_categorical_na(fwl.train.data$icu_admit_source,"Unknown")
fwl.train.data$icu_stay_type = fill_categorical_na(fwl.train.data$icu_stay_type,"Unknown")
fwl.train.data$apache_3j_bodysystem = fill_categorical_na(fwl.train.data$apache_3j_bodysystem,"Unknown")
fwl.train.data$apache_2_bodysystem = fill_categorical_na(fwl.train.data$apache_2_bodysystem,"Unknown")

# Factor convertion
fwl.train.data$hospital_death = as.factor(fwl.train.data$hospital_death) 
fwl.train.data$elective_surgery = as.factor(fwl.train.data$elective_surgery) 
fwl.train.data$aids = as.factor(fwl.train.data$aids) 
fwl.train.data$cirrhosis = as.factor(fwl.train.data$cirrhosis) 
fwl.train.data$diabetes_mellitus = as.factor(fwl.train.data$diabetes_mellitus)
fwl.train.data$hepatic_failure  = as.factor(fwl.train.data$hepatic_failure ) 
fwl.train.data$immunosuppression = as.factor(fwl.train.data$immunosuppression) 
fwl.train.data$leukemia = as.factor(fwl.train.data$leukemia) 
fwl.train.data$lymphoma = as.factor(fwl.train.data$lymphoma) 
fwl.train.data$solid_tumor_with_metastasis = as.factor(fwl.train.data$solid_tumor_with_metastasis) 

## Export to csv
write.csv(fwl.train.data,"fwl_train_data_ori.csv", row.names = T)

## Data exploration    
# Categorical Visualization
draw_categorical_chart = function(df, x){
  df %>% 
    ggplot(aes(x=x, color=hospital_death, fill=hospital_death)) +
    geom_bar(width=0.5) +
    theme_bw() +
    theme(legend.position="top")
}

draw_categorical_chart(fwl.train.data, fwl.train.data$gender)
draw_categorical_chart(fwl.train.data, fwl.train.data$elective_surgery)
draw_categorical_chart(fwl.train.data, fwl.train.data$aids)
draw_categorical_chart(fwl.train.data, fwl.train.data$cirrhosis)
draw_categorical_chart(fwl.train.data, fwl.train.data$diabetes_mellitus)
draw_categorical_chart(fwl.train.data, fwl.train.data$hepatic_falure)
draw_categorical_chart(fwl.train.data, fwl.train.data$immunosuppression)
draw_categorical_chart(fwl.train.data, fwl.train.data$leukemia)
draw_categorical_chart(fwl.train.data, fwl.train.data$lymphoma)
draw_categorical_chart(fwl.train.data, fwl.train.data$solid_tumor_with_metastasis)
levels(fwl.train.data$hepatic_failure)
count(fwl.train.data['hepatic_failure'== 0])

# Numeric Visualization

draw_numeric_chart = function(df, x){
  df %>%
    ggplot(aes(x)) +
    geom_histogram(binwidth = 10)
}

draw_numeric_chart(fwl.train.data, fwl.train.data$age)
draw_numeric_chart(fwl.train.data, fwl.train.data$bmi)
draw_numeric_chart(fwl.train.data, fwl.train.data$height)
draw_numeric_chart(fwl.train.data, fwl.train.data$pre_icu_los_days)
draw_numeric_chart(fwl.train.data, fwl.train.data$weight)
draw_numeric_chart(fwl.train.data, fwl.train.data$apache_4a_hospital_death_prob)
draw_numeric_chart(fwl.train.data, fwl.train.data$apache_4a_icu_death_prob)



