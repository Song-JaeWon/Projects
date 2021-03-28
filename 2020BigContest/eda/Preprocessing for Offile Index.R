offline = read.csv('C:/Bigcon/data/original_data/processed_data/offline_raw.csv', stringsAsFactors = F)
colnames(offline)
offline$STD_DD = as.Date(offline$STD_DD)

offline_2019 = offline[offline$YEAR == 2019, ]
offline_2019 = offline_2019[offline_2019$STD_DD > '2019-02-06',]

offline_2020 = offline[offline$YEAR == 2020, ]
offline_2020 = offline_2020[offline_2020$STD_DD > '2020-02-05',]

offline = rbind(offline_2019, offline_2020)
offline = offline %>% group_by(STD_DD, MCT_CAT_CD, YEAR, WEEK_NUM, WEEKDAY) %>% summarise(USE_CNT = sum(USE_CNT), USE_AMT = sum(USE_AMT))
offline = merge(offline, Date_class[,3:5], by = c('WEEK_NUM', 'WEEKDAY'))
offline = offline[offline$STD_DD < '2020-05-30',]

length(unique(offline$MCT_CAT_CD)) * length(unique(offline$STD_DD))
table(offline$MCT_CAT_CD)

x = offline[offline$MCT_CAT_CD == '전기', ]
y = offline[offline$MCT_CAT_CD == '의복', ]

z = merge(y[,c(1:3,5,8)], x, all.x = T)
z$MCT_CAT_CD = '전기'
z$USE_CNT = ifelse(is.na(z$USE_CNT), 1, z$USE_CNT)
z$USE_AMT = ifelse(is.na(z$USE_AMT), 1, z$USE_AMT)

offline_not_elec = offline[offline$MCT_CAT_CD != '전기',]
offline_not_elec = offline_not_elec[,c(1:3,5,8,4,6:7)]
offline = rbind(offline_not_elec, z)
offline = offline[order(offline$MCT_CAT_CD, offline$STD_DD),]
offline = offline[,c(6,3,4,1,2,5,7,8)]
rownames(offline) = NULL

write.csv(offline, 'C:/Bigcon/data/original_data/processed_data/offline_for_index.csv', row.names = F)


