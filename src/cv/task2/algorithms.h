#ifndef CGCV_ALGORITHMS_H
#define CGCV_ALGORITHMS_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

class algorithms
{
 public:
    /**
     * Datastructure for a BRIEF pixel comparison test.
     */
    typedef struct BRIEF
    {
        //! First vector for the pixel comparison test.
        cv::Point2f x;
        //! Second vector for the pixel comparison test.
        cv::Point2f y;
    } BRIEF;

    static void calcWeightedDerivative(const cv::Mat& integral_image, const int border_size,
                              const cv::KeyPoint& kp, int sigma, int s,
                              cv::Mat& weighted_derivative_x,
                              cv::Mat& weighted_derivative_y);
    static void addSurfDescriptor(const cv::Mat& integral_image, const int border_size,
                          const cv::KeyPoint& kp, cv::Mat& descriptors, int sigma, int s);
    static void calcIntegralImage(const cv::Mat& image, cv::Mat& integral_image);
    static void calcHessianResponses(const cv::Mat& image_gray,
                              const cv::Mat& integral_image,
                              const int border_size,
                              std::vector<cv::Mat>& Iyy_out,
                              std::vector<cv::Mat>& Ixx_out,
                              std::vector<cv::Mat>& Ixy_out, 
                              std::vector<cv::Mat>& response_out);
    static void calcNmsAndSurfFeatures(const cv::Mat &image_gray,
                                const cv::Mat &integral_image,
                                int border_size,
                                float threshold,
                                const std::vector<cv::Mat> &responses,
                                std::vector<cv::KeyPoint>& features,
                                cv::Mat& descriptors);
    static void calcSurfFeatures(const cv::Mat& image, const int border_size,
                          const float threshold, std::vector<cv::KeyPoint>& features,
                          cv::Mat& descriptors, cv::Mat& integral_out, 
                          std::vector<cv::Mat>& Iyy_out,
                          std::vector<cv::Mat>& Ixx_out,
                          std::vector<cv::Mat>& Ixy_out, 
                          std::vector<cv::Mat>& response_out);
    static void matchFeatures(const std::vector<cv::KeyPoint>& source_features,
                      const cv::Mat& source_descriptors,
                      const std::vector<cv::KeyPoint>& target_features,
                      const cv::Mat& target_descriptors,
                      std::vector<cv::Point2f>& good_matches_source,
                      std::vector<cv::Point2f>& good_matches_target,
                      const cv::DescriptorMatcher& matcher,
                      const float match_distance_factor);
    static cv::Mat calculateHomography(const std::vector<cv::Point2f>& good_matches_scene,
                            const std::vector<cv::Point2f>& good_matches_object,
                            int ransac_iterations,
                            float ransac_inlier_threshold,
                            std::vector<int> &inlier_indices);
    static void warpImage(const cv::Mat& img_object,
                  const cv::Mat& img_scene, const cv::Mat& img_replacement, 
                  cv::Mat& warped_replacement_img, cv::Mat& H,
                  std::vector<cv::Point2f> &scene_corners);
    static void createMask(const cv::Mat& img_scene, cv::Mat& mask, 
                  cv::Mat& warped_rep_img, cv::Mat& final_image, 
                  const std::vector<cv::Point2f> &scene_corners);
    static void calcHarrisMeasure(const cv::Mat& img, cv::Mat& R);
    static void Fast(const cv::Mat& img, std::vector<cv::KeyPoint>& features, 
              const std::vector<cv::Point> &circle, const cv::Mat& R, int N, int thresh, 
              float harrisThreshold);
    static void Brief(const cv::Mat& img, const std::vector<algorithms::BRIEF>& tests,
            const std::vector<cv::KeyPoint>& features, cv::Mat& descriptors);
};

#endif  // CGCV_ALGORITHMS_H
