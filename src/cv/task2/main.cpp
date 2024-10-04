#include <dirent.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <iomanip>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>

#include <rapidjson/document.h>

#include <sys/stat.h>

#include "algorithms.h"
#include <opencv2/opencv.hpp>
#include <opencv2/flann/random.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv_modules.hpp>

// TODO: change this if you want to implement the Bonus-Task
#define BONUS 0

#define M 256
static const int BRIEF_PATCH_SIZE = 24;

#if GENERATE_REF || FINAL_RUN
struct reference
{
  cv::Mat* mat;
  std::vector<cv::KeyPoint> *keypoints;
  std::vector<cv::Point2f> *points;
  reference(cv::Mat* m) : mat(m), keypoints(nullptr), points(nullptr) {}
  reference(std::vector<cv::KeyPoint> *kps) : mat(nullptr), keypoints(kps), points(nullptr) {}
  reference(std::vector<cv::Point2f> *pts) : mat(nullptr), keypoints(nullptr), points(pts) {}
};
#endif

/**
 * data-structure to store additional frame information in the video.
 */

void executeTestcase(rapidjson::Value&, bool bonus);

//===============================================================================
// formatNumber()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
std::string formatNumber(size_t number, size_t digits = 2)
{
  std::stringstream number_stringstream;
  number_stringstream << std::setfill('0') << std::setw(digits) << number;
  return number_stringstream.str();
}

//===============================================================================
// saveImage()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void saveImage(const std::string& out_directory, const std::string& name, cv::InputArray& image, size_t number)
{
  std::string path = out_directory + formatNumber(number) + "_" + name + ".png";
  if (image.empty() == false)
    cv::imwrite(path, image);
}

#if GENERATE_REF
//===============================================================================
// generateRef()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void generateRef(std::string path, reference ref)
{
    cv::FileStorage fs(path + ".json", cv::FileStorage::WRITE);
    if (ref.mat != nullptr)
    {
        fs << "image" << *ref.mat;
    }
    else if (ref.keypoints != nullptr)
    {
        fs << "image" << *ref.keypoints;
    }
    else if (ref.points != nullptr)
    {
        fs << "image" << *ref.points;
    }
}
#endif

#if FINAL_RUN
//===============================================================================
// getRefImage()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void getRefImage(std::string ref_directory, std::string name, reference ref)
{
    struct dirent *entry;
    DIR *dir = opendir((ref_directory).c_str());
    while ((entry = readdir(dir)) != NULL)
    {
        std::string entry_name = entry->d_name;
        if (entry_name.find(name + ".json") != std::string::npos)
        {
            std::string full_path = ref_directory + name + ".json";
            cv::FileStorage fs(full_path, cv::FileStorage::READ);
            if (ref.mat != nullptr)
            {
                fs["image"] >> *ref.mat;
            }
            else if (ref.keypoints != nullptr)
            {
                fs["image"] >> *ref.keypoints;
            }
            else if (ref.points != nullptr)
            {
                fs["image"] >> *ref.points;
            }
            break;
        }
    }
    closedir(dir);
}
#endif

void drawPointMatches(
    const cv::Mat &img_scene,
    const cv::Mat &img_object,
    const std::vector<cv::Point2f> &good_matches_source,
    const std::vector<cv::Point2f> &good_matches_target,
    cv::Mat &matched_points)
    {
    std::default_random_engine eng(42);
    std::uniform_int_distribution<int> distr(0, 255);

    matched_points = cv::Mat::zeros(img_scene.rows, img_scene.cols + img_object.cols, CV_8UC3);
    img_scene.copyTo(matched_points.colRange(0, img_scene.cols));
    cv::Mat tmp_img_object;
    cv::copyMakeBorder(img_object, tmp_img_object, 0, img_scene.rows - img_object.rows,0,0, cv::BORDER_CONSTANT);
    tmp_img_object.copyTo(matched_points.colRange(img_scene.cols, matched_points.cols));

    for (size_t z = 0; z < good_matches_source.size(); z++)
    {
        unsigned char b = distr(eng);
        unsigned char g = distr(eng);
        unsigned char r = distr(eng);
        cv::line(
            matched_points, good_matches_source[z],
            good_matches_target[z] + cv::Point2f(img_scene.cols, 0),
            cv::Scalar(b, g, r), 2);
    }
}

void getPixelsOnCircle(int radius, std::vector<cv::Point>& points)
{
    float fraction = M_PI / 56;

    int lastX = 0;
    int lastY = 0;
    for (int i = 0; i < 2 * 56; i++)
    {
        int x = round(radius * cos(i * fraction));
        int y = round(radius * sin(i * fraction));
        if (lastX != x || lastY != y)
        {
            lastX = x;
            lastY = y;
            cv::Point p(x, y);
            points.push_back(p);
        }
    }
}


void orbFeatures(const cv::Mat& img, const std::vector<algorithms::BRIEF> &tests,
                 std::vector<cv::KeyPoint>& features, cv::Mat& descriptors,
                 const std::vector<cv::Point> &circle, std::string output, size_t &image_counter)
{
    cv::Mat scaled;
    cv::Mat grey;
    cv::Mat R;
    const int N = 12;
    const int thresh = 20;
    const float harrisThreshold = 0.1;

    const float scaleFactor = 0.75f;
    cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
    float currentScaleFactor = 1.f;
    cv::Mat resultR;

    for (int scale = 0; scale < 6; scale++)
    {
        std::vector<cv::KeyPoint> scaleFeatures;
        cv::Mat scaleDescriptors;

        cv::Mat resizedImage;

        int targetWidth = currentScaleFactor * img.cols;
        int targetHeight = currentScaleFactor * img.rows;
        if (targetWidth == grey.cols && targetHeight == grey.rows)
        {
            grey.copyTo(resizedImage);
        } 
        else 
        {
            cv::resize(grey, resizedImage, cv::Size(targetWidth, targetHeight));
        }

        algorithms::calcHarrisMeasure(resizedImage, R);
        if (scale == 0)
        {
            image_counter++;
            if (!R.empty()) {
                saveImage(output, "HarrisMeasure", R*255, image_counter);
            }
        }

        if (!R.empty())
        {
            cv::Mat dilated, localMax;
            cv::dilate(R, dilated, cv::Mat());
            cv::compare(R, dilated, localMax, cv::CMP_EQ);
            localMax.convertTo(localMax, CV_32FC1);
            R = R.mul(localMax);
        }

        algorithms::Fast(resizedImage, scaleFeatures, circle, R, N, thresh, harrisThreshold);
        cv::blur(resizedImage, resizedImage, cv::Size(3, 3));
        // ommit keypoints which ware out-of-bounds...
        scaleFeatures.erase(
            std::remove_if(scaleFeatures.begin(), scaleFeatures.end(), [&resizedImage](const cv::KeyPoint &k) -> bool {
                return static_cast<int>(k.pt.x - BRIEF_PATCH_SIZE/2) < 0 || 
                       static_cast<int>(k.pt.y - BRIEF_PATCH_SIZE/2) < 0||
                       static_cast<int>(k.pt.x + BRIEF_PATCH_SIZE/2) >= resizedImage.cols ||
                       static_cast<int>(k.pt.y + BRIEF_PATCH_SIZE/2) >= resizedImage.rows;
            }), scaleFeatures.end());
        if (!scaleFeatures.empty())
            algorithms::Brief(resizedImage, tests, scaleFeatures, scaleDescriptors);

        // transform the scale features
        for (const cv::KeyPoint &kp : scaleFeatures)
        {
            features.push_back(
                cv::KeyPoint(
                    kp.pt.x/currentScaleFactor,
                    kp.pt.y/currentScaleFactor,
                    9.f/currentScaleFactor));
        }
        if (descriptors.empty())
        {
            scaleDescriptors.copyTo(descriptors);
        }
        else if (!scaleDescriptors.empty())
        {
            cv::vconcat(descriptors, scaleDescriptors, descriptors);
        }
        currentScaleFactor *= scaleFactor;
    }
}


void computeDebugDescriptorPatch(
        const cv::Mat &img_scene,
        int border_size, 
        int patch_height,
        int patch_width,
        int sigma,
        int level_size,
        cv::Mat &patch_descriptor)
{
    // dump the surf descriptor on a 30x30 center patch
    int cy = img_scene.rows / 2;
    int cx = img_scene.cols / 2;
    int start_y = cy - patch_height / 2;
    int start_x = cx - patch_width / 2;

    // make a center patch
    cv::Mat patch;
    img_scene.rowRange(start_y, start_y + patch_height).colRange(start_x, start_x + patch_width).copyTo(patch);
    cv::Mat patch_pad;
    cv::copyMakeBorder(
    patch,
        patch_pad,
        border_size,
        border_size,
        border_size,
        border_size,
        cv::BORDER_REPLICATE);
    cv::cvtColor(patch_pad, patch_pad, cv::COLOR_BGR2GRAY);

    cv::Mat int_image;
    algorithms::calcIntegralImage(patch_pad, int_image);
    cv::KeyPoint kp(patch_width/2, patch_height/2, 1.0);
    algorithms::addSurfDescriptor(int_image, border_size, kp, patch_descriptor, sigma, level_size);
}

void writeFeaturePyramid(const std::vector<cv::Mat> &feature_pyramid, 
                         const std::string &output_path, const std::string &name, size_t &image_counter)
{
    image_counter++;
    if (feature_pyramid.empty())
        return;
    int max_cols = std::max_element(
        feature_pyramid.begin(),
        feature_pyramid.end(), 
        [](const cv::Mat &a, const cv::Mat &b) -> bool { return a.cols < b.cols; })->cols;

    std::vector<int> rows;
    std::transform(
        feature_pyramid.begin(),
        feature_pyramid.end(),
        std::back_inserter(rows), [](const cv::Mat &x) -> int { return x.rows; });
    int sum_rows = std::accumulate(rows.begin(), rows.end(), 0);

    cv::Mat tmp(sum_rows, max_cols, feature_pyramid[0].type());

    int start_row_idx = 0;
    for (size_t i = 0; i < feature_pyramid.size(); i++)
    {
        int stop_row_idx = start_row_idx + feature_pyramid[i].rows;
        feature_pyramid[i].copyTo(
            tmp.rowRange(start_row_idx, stop_row_idx).
                colRange(0, feature_pyramid[i].cols));
        start_row_idx = stop_row_idx;
    }
    cv::Mat normed;
    cv::normalize(tmp, normed, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    saveImage(output_path, name, normed, image_counter);
}

std::vector<algorithms::BRIEF> generateRndPoints()
{
    std::vector<algorithms::BRIEF> tests;
    //srand(42);
    std::default_random_engine eng(42);
    std::uniform_int_distribution<int> distr(-BRIEF_PATCH_SIZE/2, BRIEF_PATCH_SIZE/2);
    for (int i = 0; i < M; i++)
    {
        //int x = rand() % 9 + 1 - 9;
        //int y = rand() % 9 + 1 - 9;
        //int x2 = rand() % 9 + 1 - 9;
        //int y2 = rand() % 9 + 1 - 9;
        int x = distr(eng);
        int y = distr(eng);
        int x2 = distr(eng);
        int y2 = distr(eng);

        algorithms::BRIEF t;
        t.x = cv::Point(x, y);
        t.y = cv::Point(x2, y2);
        tests.push_back(t);
    }
    return tests;
}

//===============================================================================
// makeDirectory()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void makeDirectory(const char* path)
{
#if defined(_WIN32)
  _mkdir(path);
#else
  mkdir(path, 0777);
#endif
}

//===============================================================================
// isPathExisting()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
bool isPathExisting(const char* path)
{
  struct stat buffer
  {
  };
  return (stat(path, &buffer)) == 0;
}

//================================================================================
// main()
//--------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//================================================================================
int main(int argc, char* argv[])
{
    std::cout << "CV/task2 framework version 1.0"
              << std::endl;  // DO NOT REMOVE THIS LINE!!!
    std::cout << "===================================" << std::endl;
    std::cout << "               CV Task 2           " << std::endl;
    std::cout << "===================================" << std::endl;

    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <config-file>" << std::endl;
        return 1;
    }

    std::ifstream fs(argv[1]);
    if (!fs)
    {
        std::cout << "Error: Failed to open file " << argv[1] << std::endl;
        return 2;
    }
    std::stringstream buffer;
    buffer << fs.rdbuf();

    rapidjson::Document doc;
    rapidjson::ParseResult check;
    check = doc.Parse<0>(buffer.str().c_str()); 

    if (check)
    {
        if (doc.HasMember("testcases"))
        {
            rapidjson::Value& testcases = doc["testcases"];
            for (rapidjson::SizeType i = 0; i < testcases.Size(); i++)
            {
                rapidjson::Value& testcase = testcases[i];  
                executeTestcase(testcase, false);

#if BONUS
                std::cout << "Starting BONUS Task..." << std::endl;
                executeTestcase(testcase, true);
#endif
            }
        }
        std::cout << "Program exited normally!" << std::endl;
    }
    else
    {
        std::cout << "Error: Failed to parse file " << argv[1] << ":"
                  << check.Offset() << std::endl;
        return 3;
    }
    return 0;
}

//==============================================================================
// executeTestcase(rapidjson::Value& testcase, bool bonus)
//------------------------------------------------------------------------------
// Executes the testcase.
//
// Parameters:
// rapidjson::Value& testcase: The json data of the testcase.
// bool bonus: TODO
//==============================================================================
void executeTestcase(rapidjson::Value& testcase, bool bonus)
{
    //========================================================================
    // Parse input data
    //========================================================================
    std::string img_object_path = testcase["img_object"].GetString();
    cv::Mat img_object = cv::imread(img_object_path);
    std::string testcase_folder = testcase["folder"].GetString();
    std::string img_scene_path = testcase["img_scene"].GetString();
    cv::Mat img_scene = cv::imread(img_scene_path);
    std::string img_rep_path = testcase["img_rep"].GetString();
    cv::Mat img_rep = cv::imread(img_rep_path);

    std::cout << "Running testcase " << testcase_folder << std::endl;

    size_t image_counter = 0;

    std::string out_directory = "./output/";
    makeDirectory(out_directory.c_str());
    out_directory.append(testcase_folder + "/");
    makeDirectory(out_directory.c_str());

    std::string ref_path = "data/intm/";
    std::string ref_directory = ref_path + testcase_folder + "/";

#if GENERATE_REF
    makeDirectory(ref_path.c_str());
    makeDirectory(ref_directory.c_str());
#endif

    if (bonus)
    {
      out_directory.append("bonus/");
      makeDirectory(out_directory.c_str());
#if GENERATE_REF || FINAL_RUN
      ref_directory.append("bonus/");
      makeDirectory(ref_directory.c_str());
#endif
    }

#if FINAL_RUN
    if (!isPathExisting(ref_directory.c_str()))
    {
      std::cout << "ref directory does not exist!" << std::endl;
      std::cout << "execute with GENERATE_REF 1 first" << std::endl;
      throw std::runtime_error("Could not load ref files");
    }
    else
    {
      std::cout << "opening ref directory" << std::endl;
    }
#endif

    //====================================================================
    // Detect the keypoints using SURF Detector
    //====================================================================
    std::cout << "Step 1 and 2 - SURF feature points, SURF descriptors... " << std::endl;

    std::vector<cv::KeyPoint> scene_features;
    cv::Mat scene_descriptors;
    cv::Mat integral_image_out;
    std::vector<cv::Mat> Iyy_out, Ixx_out, Ixy_out, response_out;
    int border_size = testcase["border"].GetInt();
    float threshold_surf = testcase["threshold"].GetDouble();
    int ransac_iterations = testcase["ransac_iterations"].GetInt();
    float ransac_inlier_threshold = testcase["ransac_inlier_threshold"].GetDouble();

    std::vector<algorithms::BRIEF> test_points = generateRndPoints();
    std::vector<cv::Point> circle;
    int radius = 6;
    getPixelsOnCircle(radius, circle);

    if (bonus)
    {
        orbFeatures(img_scene, test_points, scene_features, scene_descriptors, circle, out_directory, image_counter);
    }
    else
    {
        algorithms::calcSurfFeatures(img_scene, border_size, threshold_surf, scene_features,
                    scene_descriptors, integral_image_out, Iyy_out, Ixx_out,
                    Ixy_out, response_out);
        // dump some debugging information for students
        cv::Mat tmp;
        cv::normalize(integral_image_out, tmp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        saveImage(out_directory, "integral_image", tmp, ++image_counter);

        cv::Mat patch_descriptor;
        computeDebugDescriptorPatch(img_scene, border_size, 50, 50, 1, 3, patch_descriptor);
        computeDebugDescriptorPatch(img_scene, border_size, 50, 50, 3, 7, patch_descriptor);
        computeDebugDescriptorPatch(img_scene, border_size, 50, 50, 4, 9, patch_descriptor);

        image_counter++;
        if (patch_descriptor.rows > 0)
        {
            cv::Mat descr_normalized;
            cv::normalize(patch_descriptor, descr_normalized, 255, 0,
                        cv::NORM_MINMAX, CV_8UC1);
            saveImage(out_directory, "patch_descriptors", descr_normalized, image_counter);
        }
    }

    
    std::vector<cv::KeyPoint> object_features;
    cv::Mat object_descriptors;
    if (bonus) 
    {
        orbFeatures(img_object, test_points, object_features, object_descriptors, circle, out_directory, image_counter);
    }
    else 
    {
        algorithms::calcSurfFeatures(img_object, border_size, threshold_surf, object_features, 
                        object_descriptors, integral_image_out, Iyy_out, Ixx_out, 
                        Ixy_out, response_out);

        writeFeaturePyramid(Iyy_out, out_directory, "Iyy", image_counter);
        writeFeaturePyramid(Ixx_out, out_directory, "Ixx", image_counter);
        writeFeaturePyramid(Ixy_out, out_directory, "Ixy", image_counter);
        writeFeaturePyramid(response_out, out_directory, "hessian_response", image_counter);
    }

    
    cv::Mat feature_image_scene;
    cv::Mat feature_image_object;

    cv::drawKeypoints(img_scene, scene_features, feature_image_scene, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(img_object, object_features, feature_image_object, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    saveImage(out_directory, "features_scene", feature_image_scene, ++image_counter);
    saveImage(out_directory, "features_object", feature_image_object, ++image_counter);

#if GENERATE_REF
    generateRef(ref_directory + "scene_features", reference{&scene_features});
    generateRef(ref_directory + "object_features", reference{&object_features});
    generateRef(ref_directory + "scene_descriptors", reference{&scene_descriptors});
    generateRef(ref_directory + "object_descriptors", reference{&object_descriptors});
#endif
#if FINAL_RUN
    getRefImage(ref_directory, "scene_features", reference{&scene_features});
    getRefImage(ref_directory, "object_features", reference{&object_features});
    getRefImage(ref_directory, "scene_descriptors", reference{&scene_descriptors});
    getRefImage(ref_directory, "object_descriptors", reference{&object_descriptors});
#endif

    //====================================================================
    // Matching descriptor vectors using FLANN matcher or BFMatcher
    //====================================================================
    std::cout << "Step 3 - Calculation of correspondences..." << std::endl;
    std::vector<cv::Point2f> good_matches_scene, good_matches_object;
    
    if (bonus)
    {
        const float match_distance_factor = testcase["match_distance_factor_bonus"].GetDouble();
        cv::BFMatcher matcher(cv::NORM_HAMMING); //crossCheck = true
        algorithms::matchFeatures(scene_features, scene_descriptors, object_features,
                    object_descriptors, good_matches_scene, good_matches_object,
                    matcher, match_distance_factor);
    }
    else
    {
        const float match_distance_factor = testcase["match_distance_factor"].GetDouble();
        cvflann::seed_random(42);
        cv::FlannBasedMatcher matcher;
        algorithms::matchFeatures(scene_features, scene_descriptors, object_features,
                    object_descriptors, good_matches_scene, good_matches_object,
                    matcher, match_distance_factor);
    }


    cv::Mat matched_points;
    drawPointMatches(img_scene, img_object, good_matches_scene, good_matches_object, matched_points);
    saveImage(out_directory, "matched_features_scene", matched_points, ++image_counter);

    image_counter++;
    if (scene_descriptors.rows > 0)
    {
        cv::Mat descr_normalized;
        cv::normalize(object_descriptors, descr_normalized, 255, 0,
                    cv::NORM_MINMAX, CV_8UC1);
        saveImage(out_directory, "object_descriptors", descr_normalized, image_counter);
    }

#if GENERATE_REF
    generateRef(ref_directory + "good_matches_object", reference{&good_matches_object});
    generateRef(ref_directory + "good_matches_scene", reference{&good_matches_scene});
#endif
#if FINAL_RUN
    getRefImage(ref_directory, "good_matches_object", reference{&good_matches_object});
    getRefImage(ref_directory, "good_matches_scene", reference{&good_matches_scene});
#endif

    //====================================================================
    // Calculate homography matrix
    //====================================================================
    std::cout << "Step 4 - Calculation of homography using RANSAC..." << std::endl;

    std::vector<int> inlier_indices;

    cv::Mat H = algorithms::calculateHomography(good_matches_object, good_matches_scene, 
            ransac_iterations, ransac_inlier_threshold,
            inlier_indices);

    std::vector<cv::Point2f> inlier_points_source, inlier_points_target;
    for (int i : inlier_indices)
    {
        inlier_points_source.push_back(good_matches_scene[i]);
        inlier_points_target.push_back(good_matches_object[i]);
    }

    cv::Mat out;
    drawPointMatches(img_scene, img_object, inlier_points_source, inlier_points_target, out);
    saveImage(out_directory, "inlier_matches", out, ++image_counter);
    
#if GENERATE_REF
    generateRef(ref_directory + "H", reference{&H});
#endif
#if FINAL_RUN
    getRefImage(ref_directory, "H", reference{&H});
#endif
    
    //====================================================================
    // Superimpose
    //====================================================================
    std::cout << "Step 5 - Superimpose the transformed images..." << std::endl;
    
    cv::Mat warped_rep_img = cv::Mat::zeros(img_scene.rows, img_scene.cols, CV_8UC3);
    std::vector<cv::Point2f> scene_corners(4);
    algorithms::warpImage(img_object, img_scene, img_rep, warped_rep_img, H, scene_corners);

    cv::Mat img_scene_lines;
    img_scene.copyTo(img_scene_lines);

    cv::line( img_scene_lines, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 4 );
    cv::line( img_scene_lines, scene_corners[1], scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
    cv::line( img_scene_lines, scene_corners[2], scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
    cv::line( img_scene_lines, scene_corners[3], scene_corners[0], cv::Scalar( 0, 255, 0), 4 );
    saveImage(out_directory, "image_lines", img_scene_lines, ++image_counter);
    saveImage(out_directory, "warped_replacement", warped_rep_img, ++image_counter);
#if GENERATE_REF
    generateRef(ref_directory + "warped_replacement", reference{&warped_rep_img});
    generateRef(ref_directory + "scene_corners", reference{&scene_corners});
#endif
#if FINAL_RUN
    getRefImage(ref_directory, "warped_replacement", reference{&warped_rep_img});
    getRefImage(ref_directory, "scene_corners", reference{&scene_corners});
#endif

    // Create Mask for the overlay with help of perspectiveTransform()
    cv::Mat final_image = cv::Mat::zeros(img_scene.size(), img_scene.type());
    cv::Mat mask = cv::Mat::zeros(img_scene.rows, img_scene.cols, CV_8U);

    algorithms::createMask(img_scene, mask, warped_rep_img, 
                final_image, scene_corners);
    saveImage(out_directory, "warped_replacement_mask", mask, ++image_counter);
    saveImage(out_directory, "final_image", final_image, ++image_counter);
}
