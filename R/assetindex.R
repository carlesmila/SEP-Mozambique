#-----------------------------------------------------------------------------#
#                           1. Cleaning HDSS data                             #
#-----------------------------------------------------------------------------#

library("tidyverse")
library("ca")
library("ggrepel")

# 1. Read, select, and merge ----

# Read data
econ <- read.csv("data/HDSS/economics_sep01.csv")
locat <- read.csv("data/HDSS/location_details_sep01.csv")

# Select variables
econ <- econ[c("house_number",
               "has_telephone", "has_celular", "has_tv", "has_radio",
               "has_glacier", "has_freezer", "has_computer",
               "has_farm_of_commercial_production",
               "has_bike", "has_moto", "has_car",
               "has_cattle", "has_goat", "has_pigs")]
locat <- locat[c("house_number",
                 "wall_material", "floor_material",
                 "has_kitchen", "is_kitchen_inside", "kitchen_has_coverage",
                 "ilumination_fuel", "kitchen_fuel",
                 "is_water_src_inside", "water_source",
                 "has_latrine", "latrine_type")]

# Merge
hhdata <- inner_join(econ, locat, by = "house_number") %>%
  rename(household = house_number)
rm(econ, locat)


# 2. Clean data ----

# TV
hhdata$tv <- with(hhdata, case_when(
  has_tv==1 ~ "yes",
  has_tv==2 ~ "no"))
hhdata$has_tv <- NULL
table(hhdata$tv, useNA = "always")

# Radio
hhdata$radio <- with(hhdata, case_when(
  has_radio==1 ~ "yes",
  has_radio==2 ~ "no"))
hhdata$has_radio <- NULL
table(hhdata$radio, useNA = "always")

# Computer
hhdata$computer <- with(hhdata, case_when(
  has_computer==1 ~ "yes",
  has_computer==2 ~ "no"))
hhdata$has_computer <- NULL
table(hhdata$computer, useNA = "always")

# commercial_farm
hhdata$commercial_farm <- with(hhdata, case_when(
  has_farm_of_commercial_production==1 ~ "yes",
  has_farm_of_commercial_production==2 ~ "no"))
hhdata$has_farm_of_commercial_production <- NULL
table(hhdata$commercial_farm, useNA = "always")

# glacier_freezer
hhdata$glacier_freezer <- with(hhdata, case_when(
  has_glacier==1 | has_freezer==1 ~ "yes",
  has_glacier==2 & has_freezer==2 ~ "no"))
hhdata$has_glacier <- NULL; hhdata$has_freezer <- NULL
table(hhdata$glacier_freezer, useNA = "always")

# celular_telephone
hhdata$celular_telephone <- with(hhdata, case_when(
  has_celular == 1 | has_telephone == 1 ~ "yes",
  has_celular == 2 & has_telephone == 2 ~ "no"))
hhdata$has_celular <- NULL; hhdata$has_telephone <- NULL
table(hhdata$celular_telephone, useNA = "always")

# livestock
hhdata$livestock <- with(hhdata, case_when(
  has_pigs == 1 | has_goat == 1 | has_cattle == 1 ~ "yes",
  has_pigs == 2 & has_goat == 2 & has_cattle == 2 ~ "no"))
hhdata$has_pigs <- NULL; hhdata$has_goat <- NULL; hhdata$has_cattle <- NULL
table(hhdata$livestock, useNA = "always")

# wall
hhdata$wall <- case_when(
  hhdata$wall_material %in% 1:2 ~ "concrete/bricks",
  hhdata$wall_material %in% c(3,4,5,6,7,98) ~ "others")
hhdata$wall_material <- NULL
table(hhdata$wall, useNA = "always")

# floor
hhdata$floor <- case_when(
  hhdata$floor_material == 3 ~ "concrete",
  hhdata$floor_material %in% c(1,2,4) ~ "wood/marble/tiles",
  hhdata$floor_material %in% c(5,6,98) ~ "others")
hhdata$floor_material <- NULL
table(hhdata$floor, useNA = "always")

# kitchen_fuel
hhdata$kitchen_fuel <- case_when(
  hhdata$kitchen_fuel == 2 ~ "charcoal",
  hhdata$kitchen_fuel %in% c(1,7,98) ~ "wood/others",
  hhdata$kitchen_fuel %in% c(3,4,5) ~ "gas/electr/petrol")
table(hhdata$kitchen_fuel, useNA = "always")

# moto_car
hhdata$moto_car <- with(hhdata, case_when(
  has_moto == 1 | has_car == 1 ~ "yes",
  has_moto == 2 & has_car == 2 ~ "no"))
hhdata$has_moto <- NULL; hhdata$has_car <- NULL
table(hhdata$moto_car, useNA = "always")

# bike
hhdata$bike <- with(hhdata, case_when(
  has_bike==1 ~ "yes",
  has_bike==2 ~ "no"))
hhdata$has_bike <- NULL
table(hhdata$bike, useNA = "always")

# light_fuel
hhdata$light_fuel <- case_when(
  hhdata$ilumination_fuel %in% 1:3 ~ "electr/generator/panel",
  hhdata$ilumination_fuel %in% c(4:8,98) ~ "others")
hhdata$ilumination_fuel <- NULL
table(hhdata$light_fuel, useNA = "always")

# kitchen
hhdata$kitchen <- with(hhdata, case_when(
  has_kitchen==2 ~ "no kitchen",
  has_kitchen==1 & is_kitchen_inside==1 ~ "inside",
  has_kitchen==1 & is_kitchen_inside==2 & kitchen_has_coverage==1 ~ "outside covered",
  has_kitchen==1 & is_kitchen_inside==2  ~ "outside non-covered"
))
hhdata$has_kitchen <- NULL; hhdata$kitchen_has_coverage <- NULL; hhdata$is_kitchen_inside <- NULL;
table(hhdata$kitchen, useNA = "always")

# water
hhdata$water <- with(hhdata, case_when(
  is_water_src_inside==1 & water_source == 1 ~ "piped house",
  is_water_src_inside==1 & water_source == 2 ~ "piped compound",
  (is_water_src_inside==1 & water_source %in% c(3:8, 98)) | (is_water_src_inside==2) ~ "non-piped/outside"
))
hhdata$is_water_src_inside <- NULL; hhdata$water_source <- NULL;
table(hhdata$water, useNA = "always")

# latrine
hhdata$latrine <- with(hhdata, case_when(
  has_latrine==2 ~ "no",
  has_latrine==1 & latrine_type == 1 ~ "improved",
  has_latrine==1 & latrine_type %in% 2:4 ~ "unimproved"
))
hhdata$has_latrine <- NULL;  hhdata$latrine_type <- NULL;
table(hhdata$latrine, useNA = "always")



# 3. Missing data ----
sapply(hhdata, function(x) sum(is.na(x)))
sum(sapply(hhdata, function(x) sum(is.na(x))))/(nrow(hhdata)*(ncol(hhdata)-1)) * 100

# We impute with the mode
for(c in names(hhdata)){
  impval <- names(sort(table(hhdata[,c]),decreasing=TRUE))[1]
  hhdata[,c] <- ifelse(is.na(hhdata[,c]), impval, hhdata[,c])
}


# 4. MCA ----

# All factors, household ID as row names
row.names(hhdata) <- hhdata$household
hhdata$household <- NULL
hhdata <- mutate_all(hhdata, as.factor)

# MCA
res_mca <- mjca(hhdata, lambda = "adjusted")
# saveRDS(res_mca, "output/MCA/MCA.rds")
summary(res_mca)
plot(res_mca)

ind_mca <- data.frame(household = row.names(hhdata), mca = res_mca$rowpcoord[,1])

# Now we check with the SEP data
clean_sep_data <- read_csv("data/clean/quest_clean.csv")
all(clean_sep_data$household[!clean_sep_data$household %in% ind_mca$household])
ind_mca <- ind_mca[ind_mca$household %in% clean_sep_data$household,]

# 5. Household size ----
hhsize <- read.csv("data/HDSS/location_details_sep01.csv") |>
  select(house_number, household_size)
ind_mca <- left_join(ind_mca, hhsize, by = c("household" = "house_number"))
# 7 missing, impute with median
ind_mca$household_size <- ifelse(is.na(ind_mca$household_size),
                                 median(ind_mca$household_size, na.rm=TRUE),
                                 ind_mca$household_size)
# write_csv(ind_mca, "data/clean/asset_index.csv")


# 5. Figures ----
res_mca <- readRDS("output/MCA/MCA.rds")
plot_mca <- data.frame(var = res_mca$levelnames,
                       coord1 = res_mca$colpcoord[,1],
                       coord2 = res_mca$colpcoord[,2])
plot_mca$var <- gsub("marmor", "marble", plot_mca$var)

# Barplot of the first dimension
plot_mca <- plot_mca %>%
  arrange(coord1) %>%
  mutate(var = fct_inorder(var),
         sign = ifelse(coord1>=0, "positive", "negative"))
p <- ggplot(plot_mca) +
  geom_col(aes(x = var, y = coord1, fill = sign, col = sign), width=0.75, alpha = 0.8) +
  geom_hline(aes(yintercept = 0, alpha = 0.5)) +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle=90, hjust=1, vjust=0.5, size=10)) +
  ylab("Column principal coordinates") + xlab("") +
  coord_flip()
# ggsave("figures/mca_1st.png", p, width=6, height=8, dpi=300)
