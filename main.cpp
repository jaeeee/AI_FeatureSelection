#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace std;

// struct row {
//     int class_attribute;
//     vector<double> features;
// };

// struct data_set {
//     vector<row> rows;
// };

struct row {
    int classifier = 0;
    std::vector<double> features;
};

struct dataset {
    std::vector<row> rows;
};

struct featureSubset {
    std::vector<int> features;
    double accuracy = 0;
};

string printFeatureList(const vector<int> &v){
    string res = "";
    res += "{";
    for(int i = v.size() - 1; i >= 0; i--){
        res += to_string(v.at(i) + 1);
        if(i != 0) {
            res += ",";
        }
    }
    return res += "}";
}

dataset parseData(string fname) {
    dataset set;
    string line;
    double val;
    ifstream myfile(fname);
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            row r;
            stringstream ss;
            ss << line;
            ss >> val;
            r.classifier = val;
            while (ss >> val) {
                r.features.push_back(val);
            }
            set.rows.push_back(r);
        }
        myfile.close();
    } else {
        cout << "Unable to open file";
    }
    return set;
}

double distance(const row& r1,const row& r2,const std::vector<int>& features) {
    double sum = 0;
    for(const auto feature:features) {
        sum += (r1.features[feature] - r2.features[feature]) * (r1.features[feature] - r2.features[feature]);
    }
    return sum;
}

double leave_one_out(const dataset& data,const std::vector<int>& features) {    
    int num_correct = 0;
    for(int i = 0;i<data.rows.size();++i) {
        const auto min1_iterator = std::min_element(data.rows.begin(), data.rows.begin() + i,[&](const auto& a,const auto& b) {
            return distance(a, data.rows[i], features) < distance(b, data.rows[i], features);
        });

        const auto min2_iterator = std::min_element(data.rows.begin()+i+1, data.rows.end(), [&](const auto& a, const auto& b) {
            return distance(a, data.rows[i], features) < distance(b, data.rows[i], features);
        });

        if(i == data.rows.size()-1 ||
            (i != 0 && distance(*min1_iterator,data.rows[i],features) <= distance(*min2_iterator,data.rows[i],features))) {
            if(min1_iterator->classifier == data.rows[i].classifier) {
                ++num_correct;
            }
        }
        else if (i==0 ||
                (distance(*min1_iterator, data.rows[i], features) > distance(*min2_iterator, data.rows[i], features))) {
            if (min2_iterator->classifier == data.rows[i].classifier) {
                ++num_correct;
            }
        }
    }

    return ((double)num_correct / data.rows.size()) * 100;
    
}

std::vector<featureSubset> forwardSelection(const dataset& data) {
    std::vector<featureSubset> subsets;
    std::vector<int> current_feature_subset;
    for (int _ = 0; _ < (int)data.rows[0].features.size();++_) {
        double current_max_accuracy = -1;
        int best_feature_so_far = -1;
        for (int i = 0; i < (int)data.rows[0].features.size(); ++i) {
            //select feature i
            if (std::find(current_feature_subset.begin(), current_feature_subset.end(),i) != current_feature_subset.end()) {
                continue;
            }
            current_feature_subset.push_back(i);
            const double accuracy = leave_one_out(data, current_feature_subset);
            cout << "\tUsing feature(s) " << printFeatureList(current_feature_subset) << " accuracy is " << accuracy << "%\n";  
            if(accuracy> current_max_accuracy) {
                current_max_accuracy = accuracy;
                best_feature_so_far = i;
            }
            
            current_feature_subset.pop_back();            
        }
        current_feature_subset.push_back(best_feature_so_far);
        subsets.push_back({current_feature_subset,current_max_accuracy});
    }
    return subsets;
}

int main()
{
    string small_test_data = "CS170_SMALLtestdata__32.txt";
    string large_test_data = "CS170_largetestdata__6.txt";
    dataset parsed_data;
    int select_data;
    int select_algo;
    cout << "Enter the dataset you want to use" << endl;
    cout << "0) Small Test Data" << endl;
    cout << "1) Large Test Data" << endl;
    cin >> select_data;
    string fname;
    if (select_data == 0) {
        fname = small_test_data;
    }
    if (select_data == 1) {    
        fname = large_test_data;
    }
    parsed_data = parseData(fname);
    cout << endl;
    cout << "Type the number of the algorithm you want to run" << endl;
    cout << "1) Forward Selection" << endl;
    cout << "2) Backward Elimination" << endl;
    cin >> select_algo;
    if (select_algo == 1) {
        cout << "Running forward selection" << endl;
        auto r = forwardSelection(parsed_data);
        const auto& best_subset = *max_element(r.begin(),r.end(),[](const auto& a,const auto& b){return a.accuracy < b.accuracy;});
    }
    if (select_algo == 2) {
        cout << "Running backward elimination" << endl;
    }
    cout << "This dataset has " << parsed_data.rows.at(0).features.size() << " features (not including the class_attribute attribute), with " 
    << parsed_data.rows.size() << " instances" << endl;
    cout << "Running nearest neighbor with all " << " features, using 'leaving-one-out' evaluation, I get an accuracy of " << endl;
    return 0;
}

