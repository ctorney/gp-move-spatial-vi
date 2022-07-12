################################################################################
##                Modeling spatial-varying animal movement
##                      Majaliwa M. Masolele
################################################################################
# In this script we calculates distance from Wb locations to nearest edge,
# village,extract tourism footprint(as a metric of tourism pressure), and 
# NDVI from MODIS product available in google earth engine

rm(list = ls())

set.seed(100)
# Loading packages
package<-c("sp","rgdal","raster","tidyverse","rgeos","dplyr")
lapply(package,require,character.only=TRUE)

#Load Wb data
Wb<-read.csv("./Data/Wildebeest2019.csv")

#Define projection
latlong<-CRS(SRS_string = "EPSG:4326")
utmproj<-CRS(SRS_string = "EPSG:21036")
#latlong<-CRS("+proj=longlat +datum=WGS84 +no_defs +type=crs")

#Convert Wb data frame to spatial point dataframe
Wb<-SpatialPointsDataFrame(Wb[,c(2,3)],Wb,
                              proj4string = utmproj)

# Load the shapefiles and raster layers
sme<-readOGR("./GIS/v4_serengeti_ecosystem.shp")
Edge<-readOGR("./GIS/Sere_line.shp")
Rangerpost<-readOGR("./GIS/New ranger post.shp")
Village<-raster("./GIS/villdist.asc")
Tourism<-raster("./GIS/dist_tour.asc")

################################################################################
##                Distance calculation
################################################################################

# Check if they have the same projection if not then we need to transform
proj4string(Edge)==proj4string(Wb)

# Spatial tranformation of Wb sp point
Wb<-spTransform(Wb, CRSobj=utmproj) 
Edge<-spTransform(Edge,CRSobj=utmproj)

# Check again if they are in the same projection
proj4string(Edge)==proj4string(Wb)# They are ok

# Calculate distance to nearest Park edge(in meters)
dist.mat<-gDistance(Wb,Edge, byid=TRUE)/1000
Wb$Edge<-apply(dist.mat,2,min)

# Distance to nearest ranger post
proj4string(Rangerpost)<-CRS("+proj=utm +zone=36 +south +a=6378249.145 +rf=293.465 +units=m +no_defs +type=crs")
dist.R<-spDists(Wb,Rangerpost)/1000
Wb$Rangerpost_Dist<-apply(dist.R,1,min)
################################################################################
####                   Raster extraction                           #############
################################################################################

# Extract the Village distance values from the raster
crs(Village)<-"+proj=utm +zone=36 +south +a=6378249.145 +rf=293.465 +units=m +no_defs"
Wb<-spTransform(Wb,crs(Village))
Wb@data$Village_Dist<-raster::extract(Village,Wb,na.rm= TRUE)

# Extract the Tourism footprint from the raster
# Asigning village raster the projection
crs(Tourism)<-"+proj=utm +zone=36 +south +a=6378249.145 +rf=293.465 +units=m +no_defs"
Wb<-spTransform(Wb,crs(Tourism))
Wb@data$Tourism_fpt<-raster::extract(Tourism,Wb,na.rm= TRUE)

# Convert Wb to a dataframe
Wb<-data.frame(Wb)

################################################################################
# Extracting Time match NDVI in Google earth engine through R
################################################################################

# This script extract 16-day NDVI product from MODIS
# Matching the location and the nearest time of the wildebeest 
# Load packages
require(rgee)
require(rgeeExtra)
require(sf)
require(magick)
require(tictoc)

#Initialize the google earth engine
ee_Initialize(user="username@gmail.com")

#Setting the functions
#Function to add property with time in milliseconds
add_date<-function(feature) {
  date <- ee$Date(ee$String(feature$get("date")))$millis()
  feature$set(list(date_millis=date))
}

#Join Image and Points based on a MaxDifference Filter within a temporal window
tempwin <- 0.1 #set windows in days


maxDiffFilter<-ee$Filter$maxDifference(
  difference=tempwin*24*60*60*1000, #0.1 days * hr * min * sec * milliseconds
  leftField= "date_millis", #Date data was collected
  rightField="system:time_start" #Image date
)

# Define the join.
saveBestJoin<-ee$Join$saveBest(
  matchKey="bestImage",
  measureKey="timeDiff"
)

#Function to add property with NDVI value from the matched MODIS image
add_NDVI<-function(feature){
  #Get the best image and select NDVI
  img1<-ee$Image(feature$get("bestImage"))$select('NDVI')
  #Get ID
  id<-feature$id()
  #Extract geometry from the features
  point<-feature$geometry()
  #Get NDVI value for each point at 250 M resolution
  NDVI_value<-img1$sample(region=point, scale=250, tileScale = 16,dropNulls= FALSE)
  #Return the data containing NDVI and ids 
  feature$setMulti(list(NDVI= NDVI_value$first()$get('NDVI'), ID=id,DateTimeImage = img1$get('system:index')))
}

# Function to remove NDVI image property from features
removeProperty<- function(feature) {
  #Get the properties of the data
  properties = feature$propertyNames()
  #Select all items except images
  selectProperties = properties$filter(ee$Filter$neq("item", "bestImage"))
  #Return selected features
  feature$select(selectProperties)
}

#Add ID unique ID to Wb dataframe
Wb$No_ID<-1:nrow(Wb)

#Select column of interest and call the object el
el<-Wb[,c(2,3,4)]

#Add unique ID to el 
el$No_ID<-1:nrow(el)

# Rename the columns
colnames(el)<-c("X","Y","date","No_ID")

# Convert date to posixct
el$date<- as.POSIXct(el$date, format = "%m/%d/%Y %H:%M", tz="Africa/Nairobi") #Modify as necessary

# Convert date to factor
el$date<- as.factor(el$date)

#Put in a format that can be read by javascript
el$date<- sub(" ", "T",el$date) 

#Transform the dataframe into sf object. Make sure the name of the columns for the coordinates match. CRS needs to be in longlat WGS84.
el <- st_as_sf(el, coords = c('X','Y'), crs = 21036) 

# View the first rows of the data
head(el)

# Transform to 4326 projection
el<-st_transform(el,4326)

################################################################################
#     NDVI spatial-temporal matching and extraction
################################################################################
## Collecting image from GGE
imagecoll<-ee$ImageCollection("MODIS/006/MOD13Q1")$filterDate("1998-01-01","2019-08-30")

# Adding uniq column and replicating unique number after every 1000
el$uniq <- rep(1:1000, each=1000)[1:nrow(el)] 


# Create empty data frame to be filled with NDVI extracted values from the loop
dataoutput<- data.frame()

# Looping to extract the NDVI values
tic()
for(x in unique(el$uniq)){
  data1 <- el %>% filter(uniq == x)
  # Send sf to GEE
  data <- sf_as_ee(data1)
  # Transform day into milliseconds
  data<-data$map(add_date)
  # Apply the join.
  Data_match<-saveBestJoin$apply(data, imagecoll, maxDiffFilter)
  #Add NDVI to the data
  DataFinal<-Data_match$map(add_NDVI)
  #Remove image property from the data
  DataFinal<-DataFinal$map(removeProperty)
  # Transform GEE object in sf
  temp<- ee_as_sf(DataFinal)
  # append
  dataoutput<- rbind(dataoutput, temp)
}
toc()

#Convert to a dataframe
el_timematch<-data.frame(dataoutput)

# Multiply by 0.0001 to convert NDVI to the range of 0-1
el_timematch$NDVI<-el_timematch$NDVI*0.0001

## Add NDVI column to Wb data frame by Matching with el_timematch df
Wb$NDVI<-el_timematch$NDVI[match(Wb$No_ID,el_timematch$No_ID)]

#Save to the working directory
write.csv(Wb, file="./Data/Wildebeest2019.csv")
################################################################################
