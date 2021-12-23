#ASSIGNMENT #1 - 860

library("tidyverse")
library("sqldf")
library("readxl")

wealth <- read_excel("OneDrive - Queen's University/Global Master of Management Analytics/GMMA 860 - Acquisition and Management of Data/Assignment #1/GMMA860_Assignment1_Data.xlsx", sheet = "Wealth")
demo <- read_excel("OneDrive - Queen's University/Global Master of Management Analytics/GMMA 860 - Acquisition and Management of Data/Assignment #1/GMMA860_Assignment1_Data.xlsx", sheet = "Demo")

#PART 1

#Question A
sqldf('select PRCDDA, WSWORTHWPV from wealth order by WSWORTHWPV desc')
#ANSWER: PRCDDA = 5602, WSWORTHWPV = 2,365,581,172

#Question B
sqldf('SELECT STYAPT/(STYHOUS+STYAPT) AS APTPERCENTAGE from demo WHERE APTPERCENTAGE < .50')
#ANSWER: 5019 DAs are less than 50% condos

#Question C
sqldf('SELECT wealth.PRCDDA, demo.BASPOP, wealth.WSWORTHWPV FROM wealth JOIN demo ON wealth.PRCDDA = demo.PRCDDA order by wealth.WSWORTHWPV')
#ANSWER: ID = 190, Population = 0, Total Net Worth = 0

#Question D
#In these two datasets, wealth and demo, there would be no differences in output (on both Full Outer Join and Right Join) because of the join on ID. The ID column in both data sets has exactly 6000 unique IDs that match perfectly 1:1. In the event where demo had ID 1,2,4 and wealth had 1,2,3, there would be a difference in the output.

#Question E
sqldf('SELECT avg(WSDEBTB) FROM wealth JOIN demo ON wealth.PRCDDA = demo.PRCDDA where ACTER < 50')
#ANSWER: 11,503,957

#Part 2

library(stringr)

sales <- read_excel(file.choose())

str(sales)

sales$Product_ID <- as.character(as.numeric(sales$Product_ID))  # Convert Product_ID to character
sales$Import <- as.character(as.numeric(sales$Import))  # Convert Import to character
sales$Num_Retailers <- as.numeric(as.character(sales$Num_Retailers))  # Convert Num_Retailers to character
sales$Price <- as.numeric(gsub("\\$","",sales$Price)) #Remove $ sign and convert to numeric
sales$Product_ID <- str_pad(sales$Product_ID, 3, pad = "0") #Pad Product ID with zeros
sales1 <- gather(sales, period, total_sales, Sales_2016, Sales_2017) #Gather the data for 2016 and 2017

price_chart <- hist(sales1$Price, main = "Histogram for Price Points",xlab ="Price Points") #Create Price Point Histogram

price_plot <- ggplot(sales1, aes(y=Price, x=Num_Retailers)) + geom_point() #Create Scatterplot of price vs. number of retailers
price_plot #View the Plot

price_bar <- ggplot(sales1, aes(x=Import, y=total_sales, fill=period)) + geom_bar(stat = "identity", position=position_dodge()) #Create a sales bar chart of import vs. non-imported products
price_bar #View the Chart
