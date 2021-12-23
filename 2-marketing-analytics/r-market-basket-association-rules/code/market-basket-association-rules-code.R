library(arules)
library(data.table) ##fread
library(dplyr)
library(arulesViz)
library(plyr)
library(igraph) # get.data.frame
library(writexl)

data = read.csv(file.choose()) #use df_BBex_cleanedv2.csv (the file I shared on Teams)
summary(data)

# drop unused columns
order_dept = select(data, -c(Member, SKU, Created.On, old_date, week_no, year, Order))
order_dept_unique <- unique(order_dept)

trans1 <- as(split(order_dept_unique$Description, order_dept_unique$order_merge), "transactions")
trans1
inspect(trans1[1])
glimpse(trans1)

hist(size(trans1), breaks=0:80, xaxt="n", ylim=c(0,1000),
     main="Number of Items per Basket", xlab="# Items")
axis(1, at=seq(0,85, by=5), cex.axis=0.8)
mtext(paste("Total:", length(trans1), "baskets", sum(size(trans1)), "items"))
items_frequencies <- itemFrequency(trans1, type="a")


# APRIORI: min support = 0.005, confidence = 0.25 --- Creates 27 rules
rules <- apriori(trans1, parameter=list(support=0.005, confidence=0.25))
summary(rules)

# summary(rules) shows the following:
# parameter specification: min support, min confidence
# total number of rules
# distribution of rule length
# summary of quality measures
# information used for creating rules

inspect(rules[1:10])

# Remove redundant rule
rules <- rules[!is.redundant(rules)]
rules_dt <- data.table(lhs = labels(lhs(rules)),
                       rhs = labels(rhs(rules)),
                       quality(rules))[order(-lift),]
summary(rules_dt)
head(rules_dt, 5)


# APRIORI: min support = 0.001, confidence = 0.8, maxlen=10 --- Creates 0 rules
rules1 <- apriori(trans1, parameter=list(support=0.001, confidence=0.8, maxlen=10))
summary(rules1)
inspect(rules1[1:10])


# Item Frequency Plot
if (!require("RColorBrewer")) {
  # install color packages of RE
  install.packages("RColorBrewer")
  # include library RColorBrewer
  library(RColorBrewer)
}

# absolute plots numeric frequencies of each item independently
itemFrequencyPlot (trans1, topN=20, type="absolute",
                   col=brewer.pal(8, 'Pastel2'),
                   main="Absolute Item Frequency Plot")

# relative plots how many times items appear compared to others
itemFrequencyPlot(trans1, topN=20, type="relative",
                  col=brewer.pal(8, 'Pastel2'),
                  main="Relative Item Frequency Plot")

# scatterplot
plotly_arules(rules)
plotly_arules(rules1)

# Network graph
subrules2 <- head(sort(rules, by="confidence"), 20)
ig <- plot(subrules2, method = "graph", control=list(type="items"))
ig_df <- get.data.frame(ig, what="both")

write_xlsx(as(rules1, "data.frame"), "Overall_rules1.xlsx")

#######################################################################
################   Individual Level Example Model      ################
#######################################################################

#subset for exemplar customer
data2 = data[data$Member == 'M38622',]


# drop product columns
order_dept2 = select(data2, -c(Member, SKU, Created.On, old_date, week_no, year, Order))
order_dept_unique2 <- unique(order_dept2)

trans2 <- as(split(order_dept_unique2$Description, order_dept_unique2$order_merge), "transactions")
trans2
inspect(trans2[1])
glimpse(trans2)
#inspect(trans1)

# APRIORI: min support = 0.005, confidence = 0.25 --- Creates 28705 rules
rules2 <- apriori(trans2, parameter=list(support=0.001, confidence=0.25, maxlen=))
summary(rules2) 

# scatterplot
plotly_arules(rules2)

# APRIORI: min support = 0.001, confidence = 0.8, maxlen=10 --- Creates 29365 rules
rules3 <- apriori(trans2, parameter=list(support=0.005, confidence=0.70, maxlen=10))
summary(rules3)

write_xlsx(as(rules2, "data.frame"), "M38622_rules2.xlsx")
