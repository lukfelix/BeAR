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


#include "madhusudhan_seager_temperature.h"
#include "../additional/exceptions.h"
#include "../additional/physical_const.h"

#include <algorithm>
#include <vector>
#include <cmath>


namespace bear {


MadhusudhanSeagerTemperature::MadhusudhanSeagerTemperature()
{
  nb_parameters = 6; //total number of t-p profile parameters
}



//calculate the temperature by 3-layer solution from Madhusudhan & Seager 2019
//the temperature will be evaluated on all points given by the pressure vector
bool MadhusudhanSeagerTemperature::calcProfile(
  const std::vector<double>& parameters,
  const double surface_gravity,
  const std::vector<double>& pressure,
  std::vector<double>& temperature)
{
  temperature.assign(pressure.size(), 0);

  const double T0 = parameters[0];
  const double a1 = parameters[1];
  const double a2 = parameters[2];
  const double P1 = parameters[3];
  const double P2 = parameters[4];
  const double P3 = parameters[5];

  const double P0 = pressure[pressure.size()-1]; // TODO : check if this or first element
  const double T2 = T0 + std::pow(std::log(P1/P0)/a1, 2) - std::pow(std::log(P1/P2)/a2, 2);


  for (size_t i=0; i<pressure.size(); ++i)
  {
    if (pressure[i] >= P0 && pressure[i]<P1)
      {
        temperature[i] = T0 + std::pow(std::log(pressure[i]/P0)/a1, 2);
      }
    else if (pressure[i]>= P1 && pressure[i]<P3)
      {
        temperature[i] = T2 + std::pow(std::log(pressure[i]/P2)/a2, 2);
      }
    else if (pressure[i]>= P3)
      {
        temperature[i] = T2 + std::pow(std::log(P3/P2)/a2, 2);
      }
    }


  
  //neglect models with too low temperatures
  bool neglect_model = false;
  
  for (auto & i : temperature)
    if (i < 50) {i = 50; neglect_model = true;}


  return neglect_model;
}
}