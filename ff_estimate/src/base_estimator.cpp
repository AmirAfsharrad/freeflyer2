// MIT License
//
// Copyright (c) 2023 Stanford Autonomous Systems Lab
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#include "ff_estimate/base_estimator.hpp"

using ff_msgs::msg::FreeFlyerState;
using ff_msgs::msg::FreeFlyerStateStamped;

namespace ff
{

BaseEstimator::BaseEstimator(const std::string & node_name)
: rclcpp::Node(node_name)
{
  state_pub_ = this->create_publisher<FreeFlyerStateStamped>("est/state", 10);
}

void BaseEstimator::SendStateEstimate(const FreeFlyerState & state)
{
  ff_msgs::msg::FreeFlyerStateStamped msg{};
  msg.state = state;
  msg.header.stamp = this->get_clock()->now();

  state_pub_->publish(msg);
}

}  // namespace ff
