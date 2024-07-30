//
//  main.cpp
//  Project 4
//  Calibration and AugmentedReality
// This function shows the working of the system on press of different keys. This file enable the user to calibrate the camera and project 3D objects
//  Created by Shivani Naik and Pulkit Saharan on 04/11/22.
//

#include <iostream>
#include <string>
#include <filesystem>
#include <regex>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/calib3d.hpp>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

std::vector<cv::Point3f> readASCFile(const std::string &filename)
{
    std::vector<cv::Point3f> points;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "No se pudo abrir el archivo: " << filename << std::endl;
        return points;
    }

    double x, y, z;
    while (file >> x >> y >> z)
    {
        points.emplace_back(x, y, z);
    }

    file.close();
    return points;
}

void drawPoints(cv::Mat &image, const std::vector<cv::Point3f> &points, const cv::Mat &cameraMatrix,
                const cv::Mat &distCoeffs, const cv::Mat &rvec, const cv::Mat &tvec)
{
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    for (const auto &pt : imagePoints)
    {
        cv::circle(image, pt, 3, cv::Scalar(0, 255, 0), -1);
    }
}

void projectPointsOnBoard(cv::Mat& frame, const cv::Mat& camera_matrix, const cv::Mat& distortion_coeff, const cv::Mat& rotational_vec, const cv::Mat& trans_vec, const std::vector<cv::Point3f>& objectPoints) {
    // Verificar que las matrices sean válidas
    if (camera_matrix.empty()) {
        std::cerr << "ALERTA 1" << std::endl;
        return;
    }
    if(distortion_coeff.empty()){
        std::cerr << "ALERTA 2" << std::endl;
        return;
    }
    if(rotational_vec.empty()){
        std::cerr << "ALERTA 3" << std::endl;
        return;
    }
    if(trans_vec.empty()){
        std::cerr << "ALERTA 4" << std::endl;
        return;
    }

    // Proyectar puntos en la imagen
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(objectPoints, rotational_vec, trans_vec, camera_matrix, distortion_coeff, imagePoints);

    // Dibujar puntos proyectados en el frame
    for (const auto& point : imagePoints) {
        cv::circle(frame, point, 3, cv::Scalar(0, 0, 255), -1);
    }
}

vector<Point3f> readPoints(const string& filename) {
    vector<Point3f> points;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return points;
    }
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        float x, y, z;
        if (!(iss >> x >> y >> z)) { break; }
        points.push_back(Point3f(x, y, z));
    }
    file.close();
    return points;
}

// function splits a given string at the delimiter
std::vector<std::string> split(std::string &str, char delim)
{
    std::vector<std::string> tokens;
    std::string sub_str;

    std::stringstream str_stream(str);

    while (std::getline(str_stream, sub_str, delim))
    {
        tokens.push_back(sub_str);
    }

    return tokens;
}

// Parses an obj file and gives the vertices, face vertices, normals and face normals of the object
void parse_file(std::string f_name, vector<cv::Point3f> &vertices,
            vector<cv::Point3f> &normals, vector<std::vector<int>> &face_vertices,
               vector<int> &faceNormals)
{
    string currentLine;
    ifstream inFile(f_name);
    vector<string> tokens, indices;
    char delimiter = ' ';

    if (!inFile.is_open())
    {
        cout << "Failed to open obj file: " << f_name << endl;
        return;
    }

    // Read file line by line
    while (getline(inFile, currentLine))
    {
        // Split current line into tokens
        tokens = split(currentLine, delimiter);
        if (tokens.size() > 0)
        {
            vector<int> face_ind;
            // if line is a vertex, append to vertices
            if (tokens[0].compare("v") == 0)
            {
                vertices.push_back(cv::Point3f(std::stof(tokens[1]),std::stof(tokens[3]),(std::stof(tokens[2]))));
            }
            // if line is a vertex normal, append to normals
            else if (tokens[0].compare("vn") == 0)
            {
                normals.push_back(cv::Point3f(std::stof(tokens[1]),std::stof(tokens[2]),std::stof(tokens[3])));
            }
            // if line is a face vertex, append all vertex indices to face vertices
            else if (tokens[0].compare("f") == 0)
            {
                for(int i = 1 ; i<tokens.size() ; i++)
                {
                    // face vertex and normals are delimited by //
                    indices = split(tokens[i], '/');
                    face_ind.push_back(std::stoi(indices[0]));
                }
                face_vertices.push_back(face_ind);
                face_ind.clear();
            }
            
        }
    }
}

// Function to add chessboard coreners to corner and point list
void add_to_image_set(std::vector<cv::Point2f> corner_set, std::vector<std::vector<cv::Vec3f> >& point_list, std::vector<std::vector<cv::Point2f> >& corner_list) {


    //Defining point_set, point_list and corner_list
    std::vector<cv::Vec3f> point_set; //world co-ordinates 3D
    
    Size patternsize(9, 7);

    // add real world coordinates to point set
    for (int i = 0; i < patternsize.height; i++) {
        for (int j = 0; j < patternsize.width; j++) {
            point_set.push_back(cv::Point3f(j, -i, 0));
        }
    }

    // add point set and corner set to point and corner list
    point_list.push_back(point_set);
    corner_list.push_back(corner_set);
}

// Function to detect chessboard, extract chessboard corners and draw them
void extract_draw_corners(cv::Mat& src, std::vector<std::vector<cv::Vec3f> >& point_list, std::vector<std::vector<cv::Point2f> >& corner_list ) {

    cv::Mat gray;
    std::vector<cv::Vec3f> point_set;
    std::vector<cv::Point2f> corner_set;
    cvtColor(src, gray, cv::COLOR_RGB2GRAY);

    Size patternsize(9, 6);
    // find chessboard
    bool patternfound = findChessboardCorners(gray, patternsize, corner_set,
        CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
        + CALIB_CB_FAST_CHECK);

    if (patternfound)
        cornerSubPix(gray, corner_set, Size(11, 11), Size(-1, -1),
            TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    
    drawChessboardCorners(src, patternsize, Mat(corner_set), patternfound);
    // add corner set to point and corner list
    add_to_image_set(corner_set, point_list, corner_list);
 }

// Function creates a calibration image set from all images saved in path
void create_image_set(string path, std::vector<std::vector<cv::Vec3f> >& point_list, std::vector<std::vector<cv::Point2f> >& corner_list)
{
    for (const auto & entry : filesystem::directory_iterator(path))
    {
        string filename = entry.path().string();
        if(filename.substr(filename.length() - 3) == "png")
        {
            cout << "Processing "<<filename << endl;
            cv::Mat img = imread(filename);
            // extract and add to point and corner list
            extract_draw_corners(img, point_list, corner_list);
        }
    }
}


//Function to calibrate the camera using different images and get the intrinsic paramters
// Gives Camera matrix, distortion coefficient and calibration error
float calibrate_camera(std::vector<std::vector<cv::Vec3f>> &point_list,
                       std::vector<std::vector<cv::Point2f>> &corner_list,
                       cv::Mat &camera_matrix, cv::Mat &distortion_coeff)
{
    Size frame_size(1280,720);

    std::vector<cv::Mat> R, T;
    // Calibrate and get error
    float error = cv::calibrateCamera(point_list,
                                corner_list,
                                frame_size,
                                camera_matrix,
                                distortion_coeff,
                                R,
                                T,
                                cv::CALIB_FIX_ASPECT_RATIO,
                                cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 30, DBL_EPSILON));
    //Print outs the result from calibration

    std::cout << "cameraMatrix : " << camera_matrix << std::endl;
    std::cout << "distortion Coeffs : " << distortion_coeff << std::endl;
    std::cout << "Rotation vector : " << R[0] << std::endl;
    std::cout << "Translation vector : " << T[0] << std::endl;
    cout << "Error:" << error << std::endl;
    return(error);
}

//Function to draw 3D axes
void draw_axes(cv::Mat& src,cv::Mat &camera_matrix, cv::Mat &distortion_coeff,
              cv::Mat& R, cv::Mat& T)
{
    vector<cv::Vec3f> real_points;
    std::vector<cv::Point2f> image_points;

    // Axes coordinates in real world
    real_points.push_back(cv::Vec3f({0, 0, 0}));
    real_points.push_back(cv::Vec3f({5, 0, 0}));
    real_points.push_back(cv::Vec3f({0, -5, 0}));
    real_points.push_back(cv::Vec3f({0, 0, 5}));
    
    // get axes projections in image coordinates
    cv::projectPoints(real_points, R, T, camera_matrix, distortion_coeff, image_points);
    
    // draw axes on src image
    cv::arrowedLine(src, image_points[0], image_points[1], cv::Scalar(0, 0, 255), 3);
    cv::arrowedLine(src, image_points[0], image_points[2], cv::Scalar(0, 255, 0), 3);
    cv::arrowedLine(src, image_points[0], image_points[3], cv::Scalar(255, 0, 0), 3);
}

//Function to draw chair

void draw_chair(cv::Mat& src,cv::Mat &camera_matrix, cv::Mat &distortion_coeff,
                cv::Mat& R, cv::Mat& T)
{
    vector<cv::Vec3f> real_points;
    vector<cv::Point2f> image_points;
    Scalar color(255,255,0);
    int thickness = 2;
    
    //Chair leg squares
    real_points.push_back(cv::Vec3f({3, -2, 0}));
    real_points.push_back(cv::Vec3f({3.25, -2, 0}));
    real_points.push_back(cv::Vec3f({3.25, -2.25, 0}));
    real_points.push_back(cv::Vec3f({3, -2.25, 0}));
    
    // chair leg 1
    real_points.push_back(cv::Vec3f({3, -2, 3}));
    real_points.push_back(cv::Vec3f({3.25, -2, 3}));
    real_points.push_back(cv::Vec3f({3.25, -2.25, 3}));
    real_points.push_back(cv::Vec3f({3, -2.25, 3}));
    
    //Chair leg squares
    real_points.push_back(cv::Vec3f({3, -5, 0}));
    real_points.push_back(cv::Vec3f({3.25, -5, 0}));
    real_points.push_back(cv::Vec3f({3.25, -5.25, 0}));
    real_points.push_back(cv::Vec3f({3, -5.25, 0}));
    
    // chair leg 2
    real_points.push_back(cv::Vec3f({3, -5, 3}));
    real_points.push_back(cv::Vec3f({3.25, -5, 3}));
    real_points.push_back(cv::Vec3f({3.25, -5.25, 3}));
    real_points.push_back(cv::Vec3f({3, -5.25, 3}));
    
    //Chair leg squares
    real_points.push_back(cv::Vec3f({6, -2, 0}));
    real_points.push_back(cv::Vec3f({6.25, -2, 0}));
    real_points.push_back(cv::Vec3f({6.25, -2.25, 0}));
    real_points.push_back(cv::Vec3f({6, -2.25, 0}));
    
    // chair leg 3
    real_points.push_back(cv::Vec3f({6, -2, 3}));
    real_points.push_back(cv::Vec3f({6.25, -2, 3}));
    real_points.push_back(cv::Vec3f({6.25, -2.25, 3}));
    real_points.push_back(cv::Vec3f({6, -2.25, 3}));
    //Chair leg squares
    real_points.push_back(cv::Vec3f({6, -5, 0}));
    real_points.push_back(cv::Vec3f({6.25, -5, 0}));
    real_points.push_back(cv::Vec3f({6.25, -5.25, 0}));
    real_points.push_back(cv::Vec3f({6, -5.25, 0}));
    
    // chair leg 4
    real_points.push_back(cv::Vec3f({6, -5, 3}));
    real_points.push_back(cv::Vec3f({6.25, -5, 3}));
    real_points.push_back(cv::Vec3f({6.25, -5.25, 3}));
    real_points.push_back(cv::Vec3f({6, -5.25, 3}));
    
    //Chair base 1
    real_points.push_back(cv::Vec3f({3, -5.25, 3}));
    real_points.push_back(cv::Vec3f({6.25, -5.25, 3}));
    real_points.push_back(cv::Vec3f({6.25, -2, 3}));
    real_points.push_back(cv::Vec3f({3, -2, 3}));
    
    //Chair base 2
    real_points.push_back(cv::Vec3f({3, -5.25, 3.25}));
    real_points.push_back(cv::Vec3f({6.25, -5.25, 3.25}));
    real_points.push_back(cv::Vec3f({6.25, -2, 3.25}));
    real_points.push_back(cv::Vec3f({3, -2, 3.25}));
    
    // chair back 1
    real_points.push_back(cv::Vec3f({6.25, -2, 3.25}));
    real_points.push_back(cv::Vec3f({6.25, -2, 6}));
    real_points.push_back(cv::Vec3f({3, -2, 6}));
    real_points.push_back(cv::Vec3f({3, -2, 3.25}));
    
    // chair back 2
    real_points.push_back(cv::Vec3f({6.25, -2.25, 3.25}));
    real_points.push_back(cv::Vec3f({6.25, -2.25, 6}));
    real_points.push_back(cv::Vec3f({3, -2.25, 6}));
    real_points.push_back(cv::Vec3f({3, -2.25, 3.25}));
    
    // chair back middle line
    real_points.push_back(cv::Vec3f({4.525, -2, 3.25}));
    real_points.push_back(cv::Vec3f({4.525, -2, 6}));
    real_points.push_back(cv::Vec3f({4.725, -2, 6}));
    real_points.push_back(cv::Vec3f({4.725, -2, 3.25}));

    
    real_points.push_back(cv::Vec3f({4.525, -2.25, 3.25}));
    real_points.push_back(cv::Vec3f({4.525, -2.25, 6}));
    real_points.push_back(cv::Vec3f({4.725, -2.25, 6}));
    real_points.push_back(cv::Vec3f({4.725, -2.25, 3.25}));
    
   
    
    cv::projectPoints(real_points, R, T, camera_matrix, distortion_coeff, image_points);

    for(int i = 0; i<4;i++)
    {
        // chair leg square
        cv::line(src, image_points[8*i+0], image_points[8*i+1], color, thickness);
        cv::line(src, image_points[8*i+1], image_points[8*i+2], color, thickness);
        cv::line(src, image_points[8*i+2], image_points[8*i+3], color, thickness);
        cv::line(src, image_points[8*i+3], image_points[8*i+0], color, thickness);
        
        // chair leg
        cv::line(src, image_points[8*i+0], image_points[8*i+4], color, thickness);
        cv::line(src, image_points[8*i+1], image_points[8*i+5], color, thickness);
        cv::line(src, image_points[8*i+2], image_points[8*i+6], color, thickness);
        cv::line(src, image_points[8*i+3], image_points[8*i+7], color, thickness);
    }
    // chair base
    for(int i = 0;i<2;i++)
    {
        cv::line(src, image_points[32+0 +4*i], image_points[32+1+4*i], color, thickness);
        cv::line(src, image_points[32+1+4*i], image_points[32+2+4*i], color, thickness);
        cv::line(src, image_points[32+2+4*i], image_points[32+3+4*i], color, thickness);
        cv::line(src, image_points[32+3+4*i], image_points[32+0+4*i], color, thickness);
    }
    
    // chair base lines
    for(int i = 0; i<4;i++)
        cv::line(src, image_points[32+i], image_points[32+i+4], color, thickness);
    
    //chair back lines
    for(int i = 0;i<2;i++)
    {
        cv::line(src, image_points[40+0+4*i], image_points[40+1+4*i], color, thickness);
        cv::line(src, image_points[40+1+4*i], image_points[40+2+4*i], color, thickness);
        cv::line(src, image_points[40+2+4*i], image_points[40+3+4*i], color, thickness);
        cv::line(src, image_points[40+3+4*i], image_points[40+0+4*i], color, thickness);

        cv::line(src, image_points[40+1+i], image_points[40+5+i], color, thickness);

    }
    
    // chair back middle lines
    for(int i = 0;i<2;i++)
    {
        cv::line(src, image_points[48+0+4*i], image_points[48+1+4*i], color, thickness);
        cv::line(src, image_points[48+2+4*i], image_points[48+3+4*i], color, thickness);

    }
}

// Draw a 3d object from obj files, using parsed vertices and faces
void draw_3d_obj_object(cv::Mat &src,cv::Mat &camera_matrix, cv::Mat &distortion_coeff,
                        cv::Mat& R, cv::Mat& T, vector<cv::Point3f> vertices, vector<std::vector<int>> face_vertices)
{
    vector<cv::Point2f> image_vertices;
    Scalar color(255,0,255);
    int thickness = 1;
    
    // get image points of vertices by projecting using R, T
    cv::projectPoints(vertices, R, T, camera_matrix, distortion_coeff, image_vertices);
    
    // draw lines of every face parsed from obj file
    for(int i = 0; i<face_vertices.size(); i++)
    {
        for(int j = 0; j<face_vertices[i].size() - 1; j++)
        {
            // draw line between consecutive vertices of each face, face_vertices gives vertex index into vertex vector
            cv::line(src,image_vertices[face_vertices[i][j]-1],image_vertices[face_vertices[i][j+1]-1], color, thickness);
        }
    }

}

void draw_3d_vertices(cv::Mat &src, cv::Mat &camera_matrix, cv::Mat &distortion_coeff,
                      cv::Mat& R, cv::Mat& T, vector<cv::Point3f> vertices)
{
    vector<cv::Point2f> image_vertices;
    Scalar color(255,0,255);
    int thickness = 2; 

    cv::projectPoints(vertices, R, T, camera_matrix, distortion_coeff, image_vertices);

    for(const auto& pt : image_vertices) {
        cv::circle(src, pt, 3, color, -1); 
    }
    for(size_t i = 0; i < image_vertices.size(); i++) {
        for(size_t j = i + 1; j < image_vertices.size(); j++) {
            cv::line(src, image_vertices[i], image_vertices[j], color, thickness);
        }
    }
}

//function to draw a cube on the target
void draw_cube(cv::Mat& src, cv::Mat& camera_matrix, cv::Mat& distortion_coeff,
    cv::Mat& R, cv::Mat& T)
{
    vector<cv::Vec3f> real_points;
    vector<cv::Point2f> image_points;
    //Scalar color(255, 0, 0);
    // cube points
    real_points.push_back(cv::Vec3f({ 5, -4, 0 }));
    real_points.push_back(cv::Vec3f({ 5, -4, 2 }));
    real_points.push_back(cv::Vec3f({ 3, -4, 0 }));
    real_points.push_back(cv::Vec3f({ 3, -4, 2 }));
    real_points.push_back(cv::Vec3f({ 5, -2, 0 }));
    real_points.push_back(cv::Vec3f({ 5, -2, 2 }));
    real_points.push_back(cv::Vec3f({ 3, -2, 0 }));
    real_points.push_back(cv::Vec3f({ 3, -2, 2 }));


    cv::projectPoints(real_points, R, T, camera_matrix, distortion_coeff, image_points);
    //Lines to connect the points
    cv::line(src, image_points[0], image_points[2], cv::Scalar(0, 0, 255),2);
    cv::line(src, image_points[4], image_points[6], cv::Scalar(0, 0, 255),2);
    cv::line(src, image_points[0], image_points[4], cv::Scalar(0, 0, 255),2);
    cv::line(src, image_points[3], image_points[7], cv::Scalar(0, 0, 255),2);
    cv::line(src, image_points[2], image_points[3], cv::Scalar(0, 0, 255),2);
    cv::line(src, image_points[4], image_points[5], cv::Scalar(0, 0, 255),2);
    cv::line(src, image_points[6], image_points[7], cv::Scalar(0, 0, 255),2);
    cv::line(src, image_points[0], image_points[1], cv::Scalar(0, 0, 255),2);
    cv::line(src, image_points[2], image_points[6], cv::Scalar(0, 0, 255),2);
    cv::line(src, image_points[1], image_points[3], cv::Scalar(0, 0, 255),2);
    cv::line(src, image_points[5], image_points[7], cv::Scalar(0, 0, 255),2);
    cv::line(src, image_points[1], image_points[5], cv::Scalar(0, 0, 255),2);
    

}

//Function to draw a table
void draw_table(cv::Mat& src, cv::Mat& camera_matrix, cv::Mat& distortion_coeff, cv::Mat& R, cv::Mat& T) {

    vector<cv::Vec3f> real_points;
    vector<cv::Point2f> image_points;

    //left side leg
    real_points.push_back(cv::Vec3f({ 0, -4, 0 }));//0
    real_points.push_back(cv::Vec3f({ 0, -4, 3 }));//1
    real_points.push_back(cv::Vec3f({ 0.5, -4, 0 }));//2
    real_points.push_back(cv::Vec3f({ 0.5, -4, 3 }));//3
    real_points.push_back(cv::Vec3f({ 0, -2, 0 }));//4
    real_points.push_back(cv::Vec3f({ 0.5, -2, 0 }));//5
    real_points.push_back(cv::Vec3f({ 0, -2, 3 }));//6
    real_points.push_back(cv::Vec3f({ 0.5, -2, 3 }));//7
    //right side leg
    real_points.push_back(cv::Vec3f({ 5.5, -4, 0 }));//8
    real_points.push_back(cv::Vec3f({ 5.5, -4, 3 }));//9
    real_points.push_back(cv::Vec3f({ 6, -4, 0 }));//10
    real_points.push_back(cv::Vec3f({ 6, -4, 3 }));//11
    real_points.push_back(cv::Vec3f({5.5, -2, 0 }));//12
    real_points.push_back(cv::Vec3f({ 6, -2, 0 }));//13
    real_points.push_back(cv::Vec3f({ 5.5, -2, 3 }));//14
    real_points.push_back(cv::Vec3f({ 6, -2, 3 }));//15
    //Table top
    real_points.push_back(cv::Vec3f({ 3, -3, 3 }));//16


    cv::projectPoints(real_points, R, T, camera_matrix, distortion_coeff, image_points);
    //Drawing lines of table
    cv::line(src, image_points[0], image_points[2], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[0], image_points[1], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[0], image_points[4], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[2], image_points[3], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[2], image_points[5], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[4], image_points[5], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[4], image_points[6], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[5], image_points[7], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[1], image_points[6], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[1], image_points[3], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[3], image_points[7], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[6], image_points[7], cv::Scalar(0, 255, 255),3);

    cv::line(src, image_points[8],  image_points[10], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[8],  image_points[9],  cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[8],  image_points[12], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[10], image_points[11], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[10], image_points[13], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[12], image_points[13], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[12], image_points[14], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[13], image_points[15], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[9],  image_points[14], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[9],  image_points[11], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[11], image_points[15], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[14], image_points[15], cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[3],  image_points[9],  cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[14], image_points[7],  cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[16], image_points[3],  cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[16], image_points[9],  cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[16], image_points[7],  cv::Scalar(0, 255, 255),3);
    cv::line(src, image_points[16], image_points[14], cv::Scalar(0, 255, 255),3);

}

//Function to determine current camera position. Returns R and T matrix
bool camera_position(cv::Mat& src,cv::Mat &camera_matrix, cv::Mat &distortion_coeff,
    cv::Mat& rotational_vec, cv::Mat& trans_vec, string objType, vector<cv::Point3f> vertices, vector<std::vector<int>> face_vertices)
{

    cv::Mat gray;
    std::vector<cv::Vec3f> point_set;
    std::vector<cv::Point2f> corner_set;
    cvtColor(src, gray, cv::COLOR_RGB2GRAY);


   Size patternsize(7, 9);
    //extract corner of frame
    bool patternfound = findChessboardCorners(gray, patternsize, corner_set,
        CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
        + CALIB_CB_FAST_CHECK);


   if (patternfound)
        cornerSubPix(gray, corner_set, Size(11, 11), Size(-1, -1),
            TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));


   //extract world co-ordinates
    for (int i = 0; i < patternsize.height; i++) {
        for (int j = 0; j < patternsize.width; j++) {
            point_set.push_back(cv::Point3f(j, -i, 0));
        }
    }

    // If camera detects any pattern then draws different objects

    if (patternfound)
    {
        Scalar green(0,255,0);
        cv::solvePnP(point_set,corner_set, camera_matrix,distortion_coeff, rotational_vec,trans_vec);
        //Projects 3D axes
        if(objType == "Axes")
            draw_axes(src, camera_matrix, distortion_coeff, rotational_vec, trans_vec);
        //Projects a chair
        else if (objType == "Chair")
            draw_chair(src, camera_matrix, distortion_coeff, rotational_vec , trans_vec);
        //Projects object
        else if (objType == "Obj")
            draw_3d_obj_object(src, camera_matrix, distortion_coeff, rotational_vec , trans_vec, vertices, face_vertices);
        //Projects a cube
        else if (objType == "Cube")
            draw_cube(src, camera_matrix, distortion_coeff, rotational_vec, trans_vec);
        //Projects a table
        else if (objType == "Table")
            draw_table(src, camera_matrix, distortion_coeff, rotational_vec, trans_vec);
        else if (objType == "testing")
            projectPointsOnBoard(src, camera_matrix, distortion_coeff, rotational_vec, trans_vec, vertices);

    }

    return patternfound;
}



//Function to detect harris corner
void harris_corner(cv::Mat& src) {
    cv::Mat  src_gray;
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    //threshold value
    int thresh = 200;
    int max_thresh = 255;

    //converting image to grayscale
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    Mat dst = Mat::zeros(src_gray.size(), CV_32FC1);
    cornerHarris(src_gray, dst, blockSize, apertureSize, k);

    Mat dst_norm, dst_norm_scaled;
    Mat output = src.clone();
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    //displaying points on the image/video frame
    for (int i = 0; i < dst_norm.rows; i++)
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if ((int)dst_norm.at<float>(i, j) > thresh)
            {
                circle(output, Point(j, i), 2, Scalar(255, 0, 0), 2, 8, 0);
            }
        }
    }
    imshow("Harris_Corner", output);
}

//Aruco marker detection
bool aruco_marker_detection(cv::Mat& src, cv::Mat& output) {
    
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::aruco::DetectorParameters parameters = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector arucoDetector(dictionary,parameters);
    arucoDetector.detectMarkers(src, markerCorners, markerIds, rejectedCandidates);

   output= src.clone();
   cv::aruco::drawDetectedMarkers(output, markerCorners, markerIds);
   
   return markerCorners.size() == 4;
}

//Function to overlay an image on the ArUco markers
void aruco_out(cv::Mat& target,cv::Mat& actual, cv::Mat& output) {
    //declaration
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::aruco::DetectorParameters parameters = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector arucoDetector(dictionary,parameters);
    arucoDetector.detectMarkers(target, markerCorners, markerIds, rejectedCandidates);

   //Creating vector of size 4 to store corner points of destination image
    vector<Point2f> pts_dst(4);
    //initialize the vector with size 4
    pts_dst.resize(4);
   
  for (int i = 0;i < markerCorners.size(); i++) {
      if (markerIds[i] == 40) {
          pts_dst[0] = markerCorners[i][0];
      }
      else if (markerIds[i] == 124) {
          pts_dst[1] = markerCorners[i][0];
      }
      if (markerIds[i] == 98) {
          pts_dst[2] = markerCorners[i][1];
      }
      if (markerIds[i] == 203) {
          pts_dst[3] = markerCorners[i][3];
      }
  }
  //Creating vector of size 4 to store corner points of source image
     vector<Point2f> pts_src;
     pts_src.push_back(Point2f(0, 0));
     pts_src.push_back(Point2f(actual.size().width, 0));
     pts_src.push_back(Point2f(0, actual.size().height));
     pts_src.push_back(Point2f(actual.size().width, actual.size().height));

    //calculate homography
    cv::Mat h_matrix = cv::findHomography(pts_src, pts_dst);
   
   // Warped image
    Mat warpedImage;
    // Warp source image to destination based on homography
    warpPerspective(actual, warpedImage, h_matrix, target.size());

    output = target.clone();

   for (int i = 0; i < warpedImage.rows; i++) {
       for (int j = 0; j < warpedImage.cols; j++) {
           if (warpedImage.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
               output.at<cv::Vec3b>(i, j) = warpedImage.at<cv::Vec3b>(i, j);
           }
       }
   }

}

//Function to get the matching features of two images using ORB algorithm
void ORB_matching(cv::Mat& img_1, cv::Mat& img_2) {

    // declaring keypoint vector for both the images
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    //Creating ORB feature detector and descriptor extractor
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    //create brute force method of matching which using Hamming distance metrics
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    
    //Creating keypoints for both the images using ORB Detector
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    //Computing descriptors for both the images using ORB descriptor
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    Mat outimg1;
    //Draws keypoints in image 1
     drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);
    imshow("outimg1", outimg1);
    //imwrite("outimg1.jpg", outimg1);

    Mat outimg2;
    //Draws keypoints in image 2
    drawKeypoints(img_2, keypoints_2, outimg2, cv::Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);
    imshow("outimg2", outimg2);
    //imwrite("outimg2.jpg", outimg2);

    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);

    double min_dist = 100, max_dist = 0;
    //Creating matches between two images using brute force distance
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("Max dist : %f \n", max_dist);
    printf("Min dist : %f \n", min_dist);

    //Keeping good matches between two images using brute force distance
    std::vector< DMatch > good_matches;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (matches[i].distance <= max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match, cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0));
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch, cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0));
    imshow("img_match", img_match);
    imshow("good_match", img_goodmatch);

}








// Main function detects targets and puts the required objects in the frame on different key presses
// Command line argument: path to 3d obj file
int main(int argc, char* argv[]) {
    // Declarations

    int board_width = 7, board_height = 9, num_imgs = 50;
    cv::Size boardSize(board_width, board_height);
    float square_size = 0.036;

    cv::VideoCapture *capdev;
    cv::Mat frame, distortion_coeff, img, rotational_vec, trans_vec;
    cv::Mat output,aruco_output;
    cv::Mat src_gray;

    bool patternfound;
    string calibration_image_path = "src/calibration_images";
    cv::Mat actual = cv::imread("src/AugmentedReality/aruco.png");
    cv::Mat img_1 = cv::imread("src/AugmentedReality/eiffel1.png");
    cv::Mat img_2 = cv::imread("src/AugmentedReality/eiffel2.png");
    // command line argument for obj file
    string obj_path(argv[1]);
    
    FileStorage fs_write;
    FileStorage fs_read;
    std::vector<cv::Point2f> corner_set;
    std::vector<cv::Vec3f> point_set; //world co-ordinates 3
    std::vector<std::vector<cv::Vec3f> > point_list; // list of world co-ordinates
    std::vector<std::vector<cv::Point2f> > corner_list; //list of image points
    
    std::vector<std::vector<cv::Vec3f> > temp_point_list; // list of world co-ordinates
    std::vector<std::vector<cv::Point2f> > temp_corner_list;
    char pressed_key = 'o';
    int min_calibrate_images = 5;
    string save_path;
 
    std::vector<cv::Point3f> vertices;
    std::vector<cv::Point3f> normals;
    std::vector<std::vector<int>> face_vertices;
    std::vector<int> face_normals;
    parse_file(obj_path,vertices, normals, face_vertices, face_normals);

    
    capdev = new cv::VideoCapture(0);
    capdev->set(cv::CAP_PROP_FRAME_WIDTH, 1280);//Setting the width of the video 1280
    capdev->set(cv::CAP_PROP_FRAME_HEIGHT, 720);//Setting the height of the video// 720
    if( !capdev->isOpened() ) {
            printf("Unable to open video device\n");
            return(-1);
    }
    cv::namedWindow("Video", 1); // identifies a window

    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                   (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);
    float cols = refS.width/2;
    float rows = refS.height/2;
    
    double mat_init[3][3] = {{1,0,cols},{0,1, rows},{0,0,1}};
    
    // Create camera matrix and initialize
    cv::Mat camera_matrix = cv::Mat(3,3, CV_64FC1, &mat_init);
    cout << "Initialized camera matrix" << endl;
    // file to save camera properties
    fs_read = FileStorage("src/AugmentedReality/intrinsic.yml", FileStorage::READ);
    fs_read["camera_matrix"] >>  camera_matrix;
    fs_read["distortion_coeff"] >>  distortion_coeff;
    fs_read.release();
    cout << "Read camera matrix" << endl;
    
    while (true)
    {
        *capdev >> frame;
        if (frame.empty())
            break;

        std::vector<cv::Point2f> corners;

        bool found = cv::findChessboardCorners(frame, boardSize, corners, cv::CALIB_CB_FAST_CHECK);

        if (found)
        {
            cv::Size winSize = cv::Size(5, 5);
            cv::Size zeroZone = cv::Size(-1, -1);
            cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);

            cvtColor(frame, src_gray, cv::COLOR_BGR2GRAY);

            cv::cornerSubPix(src_gray, corners, winSize, zeroZone, criteria);

            cv::drawChessboardCorners(frame, boardSize, cv::Mat(corners), found);

            std::vector<cv::Vec3f> obj;
            for (int i = 0; i < board_height; i++)
                for (int j = 0; j < board_width; j++)
                    obj.push_back(cv::Point3f(j, -i, 0));

            std::cout << "Found corners!" << std::endl;
            corner_list.push_back(corners);
            point_list.push_back(obj);
        }
        cv::imshow("Video", frame);
        cv::waitKey(1);

        if (corner_list.size() == num_imgs)
            break;
    }
    cout<<"Calibración comenzada\n";
    calibrate_camera(point_list,corner_list,camera_matrix, distortion_coeff);
    fs_write = FileStorage("src/AugmentedReality/intrinsic.yml", FileStorage::WRITE);
    fs_write << "camera_matrix" << camera_matrix;
    fs_write << "distortion_coeff" << distortion_coeff;
    fs_write.release();

    vector<Point3f> vertices0;
    std::vector<cv::Point3f> ascPoints = readASCFile("src/PuntosLimpios50.asc");
    cv::Point3f boardCenter((board_width - 1) * square_size / 2, (board_height - 1) * square_size / 2, 0);
    std::vector<cv::Point3f> adjustedPoints;
    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if( frame.empty() ) {
          printf("frame is empty\n");
          break;
        }
    
        char key = cv::waitKey(10);
        if(key != -1)
            pressed_key = key;
        bool aruco_marker_found = false;

        std::vector<cv::Point2f> imagePoints;
        bool found = cv::findChessboardCorners(frame, boardSize, imagePoints, cv::CALIB_CB_FAST_CHECK);
        //if (found)
            //cv::drawChessboardCorners(frame, boardSize, cv::Mat(imagePoints), found);
        switch(pressed_key)
        {
            // detect and draw axes
            case 'd':
                patternfound = camera_position(frame, camera_matrix, distortion_coeff, rotational_vec, trans_vec, "Axes",vertices, face_vertices);
                if(patternfound)
                {
                    // print the vectors in real time when detects the grid
                    std::cout << "Rotation Matrix: " << std::endl;
                    for (int i = 0; i < rotational_vec.rows; i++)
                    {
                        for (int j = 0; j < rotational_vec.cols; j++)
                        {
                            std::cout << rotational_vec.at<cv::Vec2f>(i, j) << std::endl;
                        }
                    }
                    
                    std::cout << "Translation Matrix: " << std::endl;
                    for (int i = 0; i < trans_vec.rows; i++)
                    {
                        for (int j = 0; j < trans_vec.cols; j++)
                        {
                            std::cout << trans_vec.at<cv::Vec2f>(i, j) << std::endl;
                        }
                    }
                }
                cv::imshow("Video", frame);
                break;
            //Press 'h' to project chair on target
            case 'h':
                camera_position(frame, camera_matrix, distortion_coeff, rotational_vec, trans_vec, "Chair",vertices, face_vertices);
                cv::imshow("Video", frame);
                break;
            //Press '3' to project Plane on target
            case '3':
                if (found)
                    cv::drawChessboardCorners(frame, boardSize, cv::Mat(imagePoints), found);
                camera_position(frame, camera_matrix, distortion_coeff, rotational_vec, trans_vec, "Obj", vertices, face_vertices);
                cv::imshow("Video", frame);
                break;
            //Press 'b' to project Cube on target
            case 'b':
                camera_position(frame, camera_matrix, distortion_coeff, rotational_vec, trans_vec, "Cube",vertices, face_vertices);
                cv::imshow("Video", frame);
                break;
            //Press 't' to project table on target
            case 't':
                camera_position(frame, camera_matrix, distortion_coeff, rotational_vec, trans_vec, "Table",vertices, face_vertices);
                cv::imshow("Video", frame);
                break;
            //Press 'a' to detect aruco markers
            case 'a':
                aruco_marker_detection(frame, output);
                imshow("corner_image", output);
                cv::imshow("Video", frame);
                break;
            case 'l':
                camera_position(frame, camera_matrix, distortion_coeff, rotational_vec, trans_vec, "Testing",ascPoints, face_vertices);
                cv::imshow("Video", frame);
                break;
            default:
                cv::imshow("Video", frame);
                break;
        }
        // quit if key pressed is 'q'
        if( key == 'q')
            break;
    }

    return(0);
}
