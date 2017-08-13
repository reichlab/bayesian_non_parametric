library('tensorflow')



data__full <- c(589,561,640,656,727,697,640,599,568,577,553,582,600,566,653,673,742,716,660,617,583,587,565,598,628,618,688,705,770,736,678,639,604,611,594,634,658,622,709,722,782,756,702,653,615,621,602,635,677,635,736,755,811,798,735,697,661,667,645,688,713,667,762,784,837,817,767,722,681,687,660,698,717,696,775,796,858,826,783,740,701,706,677,711,734,690,785,805,871,845,801,764,725,723,690,734,750,707,807,824,886,859,819,783,740,747,711,751,804,756,860,878,942,913,869,834,790,800,763,800,826,799,890,900,961,935,894,855,809,810,766,805,821,773,883,898,957,924,881,837,784,791,760,802,828,778,889,902,969,947,908,867,815,812,773,813,834,782,892,903,966,937,896,858,817,827,797,843)
X_train <- data__full[seq(1,99,1)]
y_train <- data__full[seq(2, 100, 1)]
X_test <- X_train




batch_size <- 1            # Batch size for stochastic gradient descent
test_size <- batch_size      # Temporary heuristic. In future we'd like to decouple testing from batching
num_centr <- length(X_train)             # Number of "hidden neurons" that is number of centroids
max_iterations <- 100000       # Max number of iterations
learning_rate <- 1e-2     # Learning rate
num_classes <- 1            # Number of target classes, 10 for MNIST
var_rbf <- 225

#Obtain and proclaim sizes
N <- dim(X_train)[0]
D <- dim(X_train)[1]
Ntest <- dim(X_test)[0]

x <- tf$placeholder('float',shape=c(batch_size,D),name='input_data')
y_ <- tf$placeholder('float', shape=c(batch_size), name = 'Ground_truth')

with(tf$name_scope('input'), {
  centroids <- tf$Variable(tf$random_uniform(c(num_centr,D),dtype=tf$float32),name='centroids')

  var <- tf$Variable(tf$truncated_normal(c(num_centr),mean=20,stddev=5,dtype=tf$float32),name='RBF_variance')

  #For now, we collect the distanc
  exp_list = seq(1,length(num_centr))
  for (i in seq(1,length(num_centr))) {
        exp_list[i] =tf$exp((-1*tf$reduce_sum(tf$square(tf$subtract(x,centroids[i,])),1))/(2*var[i]))
        phi = tf$transpose(tf$stack(exp_list))
    }
})