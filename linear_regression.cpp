#include <iostream>
#include <ostream>
#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <random> 

//HOUSE PRICES (land_size, house_size, bedrooms, bathrooms, price).
Eigen::MatrixXd create_dataset(){

    Eigen::MatrixXd hi(1000,5);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution bathnbeds(1,5);
    std::uniform_int_distribution landsize(800,1200);
    std::uniform_int_distribution housesize(300,700);
    std::uniform_int_distribution noise(0,15);

    for (int x = 0; x < 1000; x++){
        int landsizee = landsize(gen);
        int bedrooms = bathnbeds(gen);
        int bathrooms = bathnbeds(gen);
        int housesizee = housesize(gen);             
        float price = landsizee * (850 + noise(gen)) + bedrooms * (20000 + noise(gen)) + bathrooms * (9000 + noise(gen)) + housesizee * (450 + noise(gen)); 
        
        int x_values[] = {landsizee, bedrooms, bathrooms, housesizee};

        for (int z = 0; z < 5; z++){
            if (z < 4){
                hi(x,z) = x_values[z];
            } else {
                hi(x,z) = price;
            } 
            
        }

    }


    return hi;
};


Eigen::MatrixXd predicty(Eigen::MatrixXd data, Eigen::MatrixXd weights){
    
    Eigen::MatrixXd output = data * weights.transpose(); 
    return output;    
};


Eigen::MatrixXd initialise_weights(Eigen::MatrixXd dataset){
    
    Eigen::MatrixXd pre_weights(dataset.rows(),dataset.cols()-1);
    
    for (int r = 0; r < dataset.rows();r++){
        Eigen::MatrixXd row = dataset.row(r);
        Eigen::MatrixXd features = dataset.leftCols(dataset.cols()-1);
        double label = row(0,row.cols()-1);
        for (int c = 0; c < features.cols(); c++){
            double W_value = label /  features(0,c) / features.cols();
            pre_weights(r,c) = W_value;
        }
    }

    Eigen::MatrixXd weights = pre_weights.colwise().sum() / pre_weights.rows();
     
    return weights;

}

Eigen::MatrixXd error(Eigen::MatrixXd predicted, Eigen::MatrixXd actual){
  
    Eigen::MatrixXd out = predicted - actual;

    return predicted - actual;

}


double return_mean_absolute_error(Eigen::MatrixXd predicted, Eigen::MatrixXd actual){
    Eigen::MatrixXd squared_diff = (predicted - actual).cwiseProduct((predicted - actual));
    double sum_of_sd = squared_diff.array().sqrt().sum();

    return sum_of_sd / squared_diff.rows();
    
}


std::vector<Eigen::MatrixXd> gradients(Eigen::MatrixXd error, Eigen::MatrixXd x, Eigen::MatrixXd weightz){

    Eigen::MatrixXd z(1,1);
    Eigen::MatrixXd y(1,1);

    Eigen::MatrixXd dW = 2 * x.transpose() * error / error.rows();

    double deeebeee = 2 * error.sum() / error.rows();
    Eigen::MatrixXd dB = Eigen::MatrixXd::Constant(error.rows(),error.cols(),deeebeee);

    std::vector<Eigen::MatrixXd> out = {dW};

    return out;
}


double print_accuracy(double mae, double avr_houseprice){
    return 100 - mae/avr_houseprice*100; 
} 


int main(){

    Eigen::MatrixXd dataset = create_dataset();
    Eigen::MatrixXd weights = initialise_weights(dataset);
    Eigen::MatrixXd x = dataset.leftCols(dataset.cols() - 1);
    Eigen::MatrixXd prediction = predicty(x,weights); 
    Eigen::MatrixXd error = prediction - dataset.col(dataset.cols()-1);

    double accuracy;
    double avr_houseprice = dataset.col(dataset.cols()-1).sum()/dataset.rows();
    double mae = return_mean_absolute_error(prediction, dataset.col(dataset.cols()-1));
    double old_mae; 
    
    int itter = 100000000;
    float alpha = 0.0000001;
    for (int m = 0; m < itter; m++){
        prediction = predicty(x,weights); 
        error = prediction - dataset.col(dataset.cols()-1);
        mae = return_mean_absolute_error(prediction, dataset.col(dataset.cols()-1));
        std::vector<Eigen::MatrixXd> gradientz = gradients(error, x, weights);

        weights -= alpha * gradientz[0].transpose();              
        if (m % 20 == 0){
            double mae = return_mean_absolute_error(prediction, dataset.col(dataset.cols()-1));
            accuracy = print_accuracy(mae,avr_houseprice); 
            std::cout << "accuracy: " << accuracy << std::endl; 
        }

        if (m % 2 == 0){
            old_mae = mae;
        }

        if (std::round(old_mae + 1) == std::round(old_mae)){
            alpha += 0.2;
        }

        if (accuracy > 99.1){
            std::cout << "\nweights that acheived this accuracy are: " << weights << std::endl;
            return 0;
        }

    }





    return 0;
}