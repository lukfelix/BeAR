/*
* This file is part of the BeAR code (https://github.com/newstrangeworlds/BeAR).
* Copyright (C) 2024 Daniel Kitzmann
*
* BeAR is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* BeAR is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You find a copy of the GNU General Public License in the main
* BeAR directory under <LICENSE>. If not, see
* <http://www.gnu.org/licenses/>.
*/


#include "twolayer_temperature.h"
#include "../additional/exceptions.h"
#include "../additional/physical_const.h"

#include <algorithm>
#include <vector>
#include <cmath>


namespace bear {


TwoLayerTemperature::TwoLayerTemperature()
{
  nb_parameters = 2; //total number of t-p profile parameters
}



//Fix temperature in atmosphere to a constant and allow surface to have different temperature
//the temperature will be evaluated on all points given by the pressure vector
bool TwoLayerTemperature::calcProfile(
  const std::vector<double>& parameters,
  const double surface_gravity,
  const std::vector<double>& pressure,
  std::vector<double>& temperature)
{

  const double Tatm = parameters[0];
  const double Tsurf = parameters[1];
  
  temperature.assign(pressure.size(), Tatm);
  temperature[0] = Tsurf;

  
  return checkProfile(temperature);
}
}