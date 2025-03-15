# **Phân tích hành vi sức khỏe con người dựa trên dữ liệu cảm biến từ thiết bị di động**  

## **Giới thiệu** 
Trong thời đại công nghệ số, các thiết bị di động như điện thoại thông minh và thiết bị đeo thông minh ngày càng trở nên phổ biến và đóng vai trò quan trọng trong việc theo dõi sức khỏe cá nhân. Các thiết bị này được trang bị nhiều cảm biến như gia tốc kế, con quay hồi chuyển và cảm biến nhịp tim, giúp thu thập một lượng lớn dữ liệu về chuyển động và hoạt động của người dùng. Việc khai thác và phân tích dữ liệu cảm biến từ thiết bị di động có thể cung cấp thông tin quan trọng về hành vi sức khỏe, hỗ trợ các ứng dụng trong lĩnh vực y tế, thể dục thể thao và theo dõi hoạt động hàng ngày.

Dự án này tập trung vào việc khai thác và phân tích dữ liệu cảm biến từ thiết bị di động nhằm phân loại các hoạt động của người dùng bằng cách sử dụng các thuật toán học máy. Cụ thể, dữ liệu từ các cảm biến sẽ được xử lý bằng Apache Spark để đảm bảo hiệu suất khi làm việc với tập dữ liệu lớn. Sau đó, các phương pháp trực quan hóa dữ liệu sẽ được thực hiện bằng R để khám phá xu hướng và đặc điểm của dữ liệu, hỗ trợ quá trình xây dựng mô hình dự đoán.

Bằng cách sử dụng mô hình Random Forest Classifier, dự án sẽ thực hiện phân loại hoạt động của người dùng dựa trên dữ liệu cảm biến. Các hoạt động có thể bao gồm đi bộ, chạy bộ, đạp xe, ngồi hoặc đứng yên. Dữ liệu đầu vào sẽ được làm sạch, tiền xử lý và chuẩn hóa để tối ưu hóa quá trình huấn luyện mô hình. Kết quả của mô hình sẽ được đánh giá dựa trên độ chính xác và hiệu suất, đồng thời so sánh với các thuật toán phân loại khác như Logistic Regression và Decision Tree để tìm ra phương pháp tối ưu nhất.

Dự án này không chỉ hướng tới mục tiêu nghiên cứu mà còn có tiềm năng ứng dụng thực tiễn, giúp phát triển các hệ thống theo dõi sức khỏe cá nhân, hỗ trợ giám sát vận động và phát hiện các hành vi bất thường. Kết quả thu được có thể đóng góp vào các giải pháp y tế thông minh, cải thiện chất lượng cuộc sống và hỗ trợ người dùng trong việc duy trì lối sống lành mạnh.

## **Mục tiêu** 
- Phân loại hoạt động người dùng: Xây dựng mô hình học máy có khả năng nhận diện và phân loại các hoạt động như đi bộ, chạy bộ, đạp xe, đứng yên hoặc ngồi dựa trên dữ liệu cảm biến.

- Nâng cao độ chính xác của mô hình: So sánh hiệu suất giữa các thuật toán khác nhau như Random Forest, Logistic Regression và Decision Tree để tìm ra phương pháp tối ưu nhất.

- Tối ưu hóa xử lý dữ liệu lớn: Sử dụng Apache Spark để cải thiện hiệu suất xử lý dữ liệu lớn, giúp hệ thống có thể mở rộng và xử lý tập dữ liệu lớn trong thời gian ngắn.

- Trực quan hóa và phân tích dữ liệu: Hiển thị các xu hướng và đặc điểm của dữ liệu bằng các biểu đồ trực quan như Histogram, Boxplot, Scatter Plot và Heatmap.

- Ứng dụng vào thực tế: Mở rộng khả năng ứng dụng của mô hình vào các lĩnh vực như giám sát sức khỏe, hỗ trợ vận động viên, phát hiện hành vi bất thường hoặc nguy cơ té ngã ở người cao tuổi.
## **Cài đặt** 
**1. Cài đặt Apache Spark**
Chạy lệnh sau trong R:
```
install.packages("sparklyr")
sparklyr::spark_install()
```
**2. Cài đặt các gói R cần thiết**
```
install.packages(c("sparklyr", "dplyr", "ggplot2", "reshape2", "gridExtra"))
```
## **Hướng dẫn thực hiện**
## Phần 1: Tạo biểu đồ trực quan hóa
**1.1 Kết nối với Spark**
```
sc <- spark_connect(master = "local")
```
**1.2. Đọc dữ liệu CSV vào Spark**
```
mhealth <- spark_read_csv(sc,
                          name = "mhealth",
                          path = "C:/mhealth_raw_data.csv",  # Thay đổi đường dẫn file CSV cho đúng
                          header = TRUE,
                          infer_schema = TRUE)
```
**1.3. Thu thập dữ liệu về R để phân tích trực quan (EDA)**
```
mhealth_local <- mhealth %>% collect()
cat("Số dòng dữ liệu gốc:", nrow(mhealth_local), "\n")
```
- Kết quả
```
Số dòng dữ liệu gốc: 1215745
```
- Nếu dữ liệu quá lớn, lấy mẫu (ví dụ 50.000 dòng) để giảm tải
```
if(nrow(mhealth_local) > 50000) {
  set.seed(123)
  mhealth_sample <- mhealth_local %>% sample_n(50000)
  cat("Số dòng sau khi lấy mẫu:", nrow(mhealth_sample), "\n")
} else {
  mhealth_sample <- mhealth_local
}
```
- Kết quả
```
Số dòng sau khi lấy mẫu: 50000
```
**1.4. Khai báo các cột cảm biến**
```
sensor_columns <- c("alx", "aly", "alz", 
                    "glx", "gly", "glz", 
                    "arx", "ary", "arz", 
                    "grx", "gry", "grz")
```
- Chuyển "Activity" và "subject" thành factor nếu là biến phân loại
```
mhealth_sample$Activity <- as.factor(mhealth_sample$Activity)
mhealth_sample$subject  <- as.factor(mhealth_sample$subject)
```
**1.5. Tạo các biểu đồ EDA**

**1.5.1. Histogram: Phân bố giá trị alx**
```
p1 <- ggplot(mhealth_sample, aes(x = alx)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  labs(title = "Phân bố giá trị alx", x = "alx", y = "Tần suất") +
  theme_minimal()
```
![image](https://github.com/user-attachments/assets/744339aa-575a-464c-8956-e5b786afc44a)

**1.5.2. Boxplot: So sánh alx theo Activity**
```
p2 <- ggplot(mhealth_sample, aes(x = Activity, y = alx, fill = Activity)) +
  geom_boxplot() +
  labs(title = "Boxplot alx theo Activity", x = "Activity", y = "alx") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```
![image](https://github.com/user-attachments/assets/6a1d7aca-f9b0-496c-b705-357a5cd5b429)

**1.5.3. Scatter Plot: Mối quan hệ giữa alx và aly**
```
p3 <- ggplot(mhealth_sample, aes(x = alx, y = aly)) +
  geom_point(alpha = 0.4, color = "red") +
  labs(title = "Mối quan hệ giữa alx và aly", x = "alx", y = "aly") +
  theme_minimal()
```
![image](https://github.com/user-attachments/assets/3dbfbf9b-a781-4cea-9634-40d0683efce9)

**1.5.4. Heatmap: Ma trận tương quan các cảm biến**
```
sensor_data <- mhealth_sample %>% select(all_of(sensor_columns))
cor_matrix <- cor(sensor_data, use = "complete.obs")
melted_cor <- melt(cor_matrix)

p4 <- ggplot(melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  labs(title = "Ma trận tương quan các cảm biến", x = "Cảm biến", y = "Cảm biến") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```
![image](https://github.com/user-attachments/assets/aeb9a5a3-05aa-4c0a-a98e-1d6a7eba3d25)

**1.5.5. Density Plot: Đường mật độ của alx**
```
p5 <- ggplot(mhealth_sample, aes(x = alx)) +
  geom_density(fill = "purple", alpha = 0.5) +
  labs(title = "Đường mật độ của alx", x = "alx", y = "Mật độ") +
  theme_minimal()
```
![image](https://github.com/user-attachments/assets/a2d0e779-c072-4562-ab65-97aef4f92126)

**1.5.6 Hiển thị các biểu đồ**
```
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)
```

## Phần 2: Áp dụng các thuật toán học máy trên dữ liệu Spark

**2.1. Huấn luyện mô hình Random Forest Classifier**
```
rf_model <- training %>% 
  ml_random_forest_classifier(response = "label", features = "features_vec", num_trees = 50)
predictions_rf <- ml_predict(rf_model, test)
accuracy_rf <- predictions_rf %>% 
  ml_multiclass_classification_evaluator(label_col = "label", prediction_col = "prediction")
cat("Độ chính xác của Random Forest:", accuracy_rf, "\n")
```
Kết quả
```
Độ chính xác của Random Forest: 0.6031886
```
**2.2. Huấn luyện mô hình Logistic Regression**
```
lr_model <- training %>% 
  ml_logistic_regression(response = "label", features = "features_vec")
predictions_lr <- ml_predict(lr_model, test)
accuracy_lr <- predictions_lr %>% 
  ml_multiclass_classification_evaluator(label_col = "label", prediction_col = "prediction")
cat("Độ chính xác của Logistic Regression:", accuracy_lr, "\n")
```
Kết quả
```
Độ chính xác của Logistic Regression: 0.6007053
```
**2.3. Huấn luyện mô hình Decision Tree Classifier**
```
dt_model <- training %>% 
  ml_decision_tree_classifier(response = "label", features = "features_vec")
predictions_dt <- ml_predict(dt_model, test)
accuracy_dt <- predictions_dt %>% 
  ml_multiclass_classification_evaluator(label_col = "label", prediction_col = "prediction")
cat("Độ chính xác của Decision Tree:", accuracy_dt, "\n")
```
Kết quả
```
Độ chính xác của Decision Tree: 0.6045617
```
**2.4. (Tùy chọn) Phân cụm với K-means (unsupervised)**
```
kmeans_model <- ml_kmeans(training, features = "features_vec", k = 3)
predictions_kmeans <- ml_predict(kmeans_model, test)
cat("Đã thực hiện K-means clustering với k = 3.\n")
```
Kết quả
```
Đã thực hiện K-means clustering với k = 3.
```
## **Ngắt kết nối Spark**
```
spark_disconnect(sc)
```
## **Hướng dẫn chạy mã** 
- Bước 1: Ấn Ctrl+A để bôi đen toàn bộ code
- Bước 2: Ẩn Ctrl+Enter để chạy toàn bộ code
