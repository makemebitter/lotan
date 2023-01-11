#!/usr/bin/env bash
# Copyright 2023 Yuhao Zhang and Arun Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

export DEBIAN_FRONTEND=noninteractive
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo add-apt-repository multiverse
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa

sudo apt-get install -y $(cat pkglist)


# setup sysstat
sudo sed -i 's/"false"/"true"/g' /etc/default/sysstat
sudo service sysstat restart

# java
sudo apt-get install -y openjdk-8-jdk
sudo update-java-alternatives --set /usr/lib/jvm/java-1.8.0-openjdk-amd64


# # blas
# install_mkl
sudo apt-get install -y intel-mkl-full
sudo ln -sf /usr/lib/x86_64-linux-gnu/mkl/libmkl_rt.so /usr/local/lib/liblapack.so.3
sudo ln -sf /usr/lib/x86_64-linux-gnu/mkl/libmkl_rt.so /usr/local/lib/libblas.so.3
sudo update-alternatives --set libblas.so.3-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/libmkl_rt.so
sudo update-alternatives --set liblapack.so.3-x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/libmkl_rt.so

sudo update-alternatives --install /usr/lib/liblapack.so.3 liblapack.so.3 /usr/lib/x86_64-linux-gnu/libmkl_rt.so 1000
sudo update-alternatives --install /usr/lib/liblapack.so liblapack.so /usr/lib/x86_64-linux-gnu/libmkl_rt.so 1000
sudo update-alternatives --install /usr/lib/libblas.so.3 libblas.so.3 /usr/lib/x86_64-linux-gnu/libmkl_rt.so 1000
sudo update-alternatives --install /usr/lib/libblas.so libblas.so /usr/lib/x86_64-linux-gnu/libmkl_rt.so 1000

sudo apt-get install -y msttcorefonts -qq
sudo ldconfig