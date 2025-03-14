# -------------------------------
# Phần 1: Khai thác & Trực quan hóa Dữ liệu (EDA)
# -------------------------------

# Cài đặt (nếu chưa có) và nạp các gói cần thiết
# install.packages("sparklyr")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("reshape2")
# install.packages("gridExtra")

library(sparklyr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(gridExtra)

# 1. Kết nối với Spark
sc <- spark_connect(master = "local")

# 2. Đọc dữ liệu CSV vào Spark
mhealth <- spark_read_csv(sc,
                          name = "mhealth",
                          path = "C:/mhealth_raw_data.csv",  # Thay đổi đường dẫn file CSV cho đúng
                          header = TRUE,
                          infer_schema = TRUE)

# 3. Thu thập dữ liệu về R để phân tích trực quan (EDA)
mhealth_local <- mhealth %>% collect()
cat("Số dòng dữ liệu gốc:", nrow(mhealth_local), "\n")

# Nếu dữ liệu quá lớn, lấy mẫu (ví dụ 50.000 dòng) để giảm tải
if(nrow(mhealth_local) > 50000) {
  set.seed(123)
  mhealth_sample <- mhealth_local %>% sample_n(50000)
  cat("Số dòng sau khi lấy mẫu:", nrow(mhealth_sample), "\n")
} else {
  mhealth_sample <- mhealth_local
}

# Kiểm tra tên cột
print(colnames(mhealth_sample))
# Dữ liệu dự kiến gồm: "alx", "aly", "alz", "glx", "gly", "glz", "arx", "ary", "arz", "grx", "gry", "grz", "Activity", "subject"

# 4. Khai báo các cột cảm biến
sensor_columns <- c("alx", "aly", "alz", 
                    "glx", "gly", "glz", 
                    "arx", "ary", "arz", 
                    "grx", "gry", "grz")

# Chuyển "Activity" và "subject" thành factor nếu là biến phân loại
mhealth_sample$Activity <- as.factor(mhealth_sample$Activity)
mhealth_sample$subject  <- as.factor(mhealth_sample$subject)

# 5. Tạo các biểu đồ EDA

## 5.1. Histogram: Phân bố giá trị alx
p1 <- ggplot(mhealth_sample, aes(x = alx)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  labs(title = "Phân bố giá trị alx", x = "alx", y = "Tần suất") +
  theme_minimal()

## 5.2. Boxplot: So sánh alx theo Activity
p2 <- ggplot(mhealth_sample, aes(x = Activity, y = alx, fill = Activity)) +
  geom_boxplot() +
  labs(title = "Boxplot alx theo Activity", x = "Activity", y = "alx") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

## 5.3. Scatter Plot: Mối quan hệ giữa alx và aly
p3 <- ggplot(mhealth_sample, aes(x = alx, y = aly)) +
  geom_point(alpha = 0.4, color = "red") +
  labs(title = "Mối quan hệ giữa alx và aly", x = "alx", y = "aly") +
  theme_minimal()

## 5.4. Heatmap: Ma trận tương quan các cảm biến
sensor_data <- mhealth_sample %>% select(all_of(sensor_columns))
cor_matrix <- cor(sensor_data, use = "complete.obs")
melted_cor <- melt(cor_matrix)

p4 <- ggplot(melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  labs(title = "Ma trận tương quan các cảm biến", x = "Cảm biến", y = "Cảm biến") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

## 5.5. Density Plot: Đường mật độ của alx
p5 <- ggplot(mhealth_sample, aes(x = alx)) +
  geom_density(fill = "purple", alpha = 0.5) +
  labs(title = "Đường mật độ của alx", x = "alx", y = "Mật độ") +
  theme_minimal()

# Hiển thị các biểu đồ
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)

# Lưu biểu đồ ra file PNG
ggsave("hist_alx.png", p1, width = 6, height = 4, dpi = 100)
ggsave("boxplot_alx_activity.png", p2, width = 6, height = 4, dpi = 100)
ggsave("scatter_alx_aly.png", p3, width = 6, height = 4, dpi = 100)
ggsave("heatmap_sensors.png", p4, width = 6, height = 4, dpi = 100)
ggsave("density_alx.png", p5, width = 6, height = 4, dpi = 100)

# -------------------------------
# Phần 2: Áp dụng các thuật toán học máy trên dữ liệu Spark
# -------------------------------

# Sử dụng dữ liệu gốc trên Spark (mhealth) để huấn luyện mô hình

# Chia dữ liệu trong Spark thành tập huấn luyện (70%) và tập kiểm tra (30%)
partitions <- mhealth %>% sdf_random_split(training = 0.7, test = 0.3, seed = 123)
training <- partitions$training
test <- partitions$test

# Chuyển đổi cột Activity thành nhãn số (label) sử dụng ft_string_indexer
training <- training %>% ft_string_indexer(input_col = "Activity", output_col = "label")
test <- test %>% ft_string_indexer(input_col = "Activity", output_col = "label")

# Kiểm tra và loại bỏ cột "features_vec" nếu đã tồn tại để tránh lỗi trùng lặp
if("features_vec" %in% colnames(training)) {
  training <- training %>% select(-features_vec)
}
if("features_vec" %in% colnames(test)) {
  test <- test %>% select(-features_vec)
}

# Tổng hợp các cột cảm biến thành vector đặc trưng, đổi tên output thành "features_vec"
training <- training %>% ft_vector_assembler(input_cols = sensor_columns, output_col = "features_vec")
test <- test %>% ft_vector_assembler(input_cols = sensor_columns, output_col = "features_vec")

### 2.1. Huấn luyện mô hình Random Forest Classifier
rf_model <- training %>% 
  ml_random_forest_classifier(response = "label", features = "features_vec", num_trees = 50)
predictions_rf <- ml_predict(rf_model, test)
accuracy_rf <- predictions_rf %>% 
  ml_multiclass_classification_evaluator(label_col = "label", prediction_col = "prediction")
cat("Độ chính xác của Random Forest:", accuracy_rf, "\n")

### 2.2. Huấn luyện mô hình Logistic Regression
lr_model <- training %>% 
  ml_logistic_regression(response = "label", features = "features_vec")
predictions_lr <- ml_predict(lr_model, test)
accuracy_lr <- predictions_lr %>% 
  ml_multiclass_classification_evaluator(label_col = "label", prediction_col = "prediction")
cat("Độ chính xác của Logistic Regression:", accuracy_lr, "\n")

### 2.3. Huấn luyện mô hình Decision Tree Classifier
dt_model <- training %>% 
  ml_decision_tree_classifier(response = "label", features = "features_vec")
predictions_dt <- ml_predict(dt_model, test)
accuracy_dt <- predictions_dt %>% 
  ml_multiclass_classification_evaluator(label_col = "label", prediction_col = "prediction")
cat("Độ chính xác của Decision Tree:", accuracy_dt, "\n")

### 2.4. Huấn luyện mô hình Gradient Boosted Trees
gbt_model <- training %>% 
  ml_gradient_boosted_trees(response = "label", features = "features_vec")
predictions_gbt <- ml_predict(gbt_model, test)
accuracy_gbt <- predictions_gbt %>% 
  ml_multiclass_classification_evaluator(label_col = "label", prediction_col = "prediction")
cat("Độ chính xác của Gradient Boosted Trees:", accuracy_gbt, "\n")

### 2.5. (Tùy chọn) Giảm chiều dữ liệu bằng PCA

# Chuyển đổi cột vector đặc trưng "features_vec" thành dạng dense
training <- training %>% ft_densify(input_col = "features_vec", output_col = "dense_features")
test <- test %>% ft_densify(input_col = "features_vec", output_col = "dense_features")

# Áp dụng PCA trên cột dense_features với k = 5 và lưu kết quả vào cột "pca_features"
pca_model <- training %>% ml_pca(features = "dense_features", k = 5, output_col = "pca_features")
training <- ml_transform(pca_model, training)
test <- ml_transform(pca_model, test)
cat("Đã áp dụng PCA để giảm chiều dữ liệu.\n")


### 2.6. (Tùy chọn) Phân cụm với K-means (unsupervised)
kmeans_model <- ml_kmeans(training, features = "features_vec", k = 3)
predictions_kmeans <- ml_predict(kmeans_model, test)
cat("Đã thực hiện K-means clustering với k = 3.\n")

# -------------------------------
# Ngắt kết nối với Spark
# -------------------------------
spark_disconnect(sc)
