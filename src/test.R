library("diffusionMap")
nc <= 3
nr <- 20
x<- runif(nr*nc,min=0.01,max=1)
m <- matrix(x,nrow=20,ncol=3)
xy <- dist(m)

ans <- diffuse(xy)
dum <- nystrom(ans,xy,xy)
dum
