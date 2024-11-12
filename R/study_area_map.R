#-----------------------------------------------------------------------------#
#                              Study area figure                              #
#-----------------------------------------------------------------------------#

library("sf")
library("tmap")
library("dplyr")
library("cowplot")

# Read data ----
countries <- read_sf("data/map/countries.gpkg")
moz <- countries[countries$NAME_EN=="Mozambique",]
provinces <- read_sf("data/map/provinces.gpkg")
manhica <- read_sf("data/map/manhica.gpkg")
postos <- read_sf("data/map/postos.gpkg")
sede <- postos[postos$Admin_Post=="Manhiça Sede",]
urban <- read_sf("data/map/urban.gpkg")
roads <- read_sf("data/map/roads.gpkg")
buff <- st_buffer(st_as_sfc(st_bbox(manhica)), 10000)

# Map 1: Outer view
m1 <- tm_shape(countries, bbox = moz) +
  tm_polygons(col = "#ffffff", border.col = "#3A3A3A", lwd = 0.5) +
tm_shape(provinces) +
  tm_polygons(col = "#ebebeb", border.col = "#3A3A3A", lwd = 0.5) +
tm_shape(manhica) +
  tm_polygons(col = "#2C2C2C", border.col = "#2C2C2C", lwd = 1) +
tm_shape(buff) +
  tm_borders(col = "purple", lwd = 3) +
tm_layout(bg.color = "#AFDCF0")

# Map 2: Inner view
m2 <- tm_shape(countries, bbox = manhica) +
  tm_polygons(col = "#ffffff", lwd = 0) +
tm_shape(manhica) +
  tm_polygons(col = "#ebebeb", border.col = "#3A3A3A", lwd = 1.5) +
tm_shape(postos) +
  tm_polygons(col = "#ebebeb", border.col = "#3A3A3A", lwd = 0.25) +
tm_shape(sede) +
  tm_polygons(col = "grey30", border.col = "#3A3A3A", lwd = 0.25) +
  tm_text("Admin_Post") +
tm_shape(roads) +
  tm_lines(col = "gold", lwd = 2)  +
tm_shape(urban) +
  tm_fill(col = "#FF3C39") +
tm_graticules(lwd = 0.3, labels.size = 0.6, alpha=0.75) +
tm_add_legend(type = "fill", col = "#ebebeb", border.col = "#3A3A3A", lwd = 1.5, label = "Manhiça district") +
tm_add_legend(type = "symbol", size = 0.4, col = "#FF3C39", border.col = "#FF3C39", shape = 22, label = "Urban area") +
tm_add_legend(type = "line", col = "gold", lwd = 2, label = "Main road") +
tm_scale_bar(text.size = 0.6) +
tm_layout(bg.color = "#AFDCF0", legend.outside = T, legend.text.size = 0.75,
          legend.outside.size = 0.2)

# Composite
m1_grob <- tmap_grob(m1)
m2_grob <- tmap_grob(m2)
im <- ggdraw() +
  draw_plot(m2_grob) +
  draw_plot(m1_grob,
            height = 0.4,
            vjust = -0.11, hjust = -0.385)
# png("figures/study_area.png", res = 300*4, width = 2200*4, height = 1800*4)
# im
# dev.off()
