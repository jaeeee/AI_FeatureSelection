#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

string small_test_data = "CS170_SMALLtestdata__32.txt";
string large_test_data = "CS170_largetestdata__6.txt";

struct row {
    int class_attribute;
    vector<double> features;
};

struct data_set {
    vector<row> rows;
};


data_set parseData(string fname) {
    data_set set;
    row r;
    string line;
    double val;
    ifstream myfile(fname);
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            r.features.clear();
            stringstream ss;
            ss << line;
            ss >> val;
            r.class_attribute = val;
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

int main()
{
    data_set parsed_data;
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
    }
    if (select_algo == 2) {
        cout << "Running backward elemination" << endl;
    }
    cout << "This dataset has " << parsed_data.rows.at(0).features.size() << " features (not including the class_attribute attribute), with " 
    << parsed_data.rows.size() << "instances" << endl;
    cout << "Running nearest neighbor with all " << " features, using 'leaving-one-out' evaluation, I get an accuracy of " << endl;
    return 0;
}

