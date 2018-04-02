/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    //initialize number of particles
    num_particles = 150;
    
    //create a normal gaussian distribution for x, y and theta
    normal_distribution<double> dist_x(x,std[0]);
    normal_distribution<double> dist_y(y,std[1]);
    normal_distribution<double> dist_theta(theta,std[2]);
    
    //sample from the normal distributions
    default_random_engine gen;
    
    //for all particles
    for(int i=0; i < num_particles; i++){
        
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        
        particles.push_back(p);
        weights.push_back(1.0);
    }
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    default_random_engine gen;
    
    //create a normal gaussian distribution for x, y and theta
    normal_distribution<double> noise_x(0,std_pos[0]);
    normal_distribution<double> noise_y(0, std_pos[1]);
    normal_distribution<double> noise_theta(0, std_pos[2]);
    
    //for all particles
    for(int i=0; i < num_particles; i++){

        //if yaw rate is not 0
        if(fabs(yaw_rate) < 0.00001){
            //find new position of the vehicle
            particles[i].x += velocity*delta_t*cos(particles[i].theta);
            particles[i].y += velocity*delta_t*sin(particles[i].theta);
        }
        
        else{
            //find new position of the vehicle
            particles[i].x += velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t)-sin(particles[i].theta));
            particles[i].y += velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
            particles[i].theta += yaw_rate*delta_t;
        }
        
        //add noise
        particles[i].x += noise_x(gen);
        particles[i].y += noise_y(gen);
        particles[i].theta += noise_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    //for all observations
    for(int i=0; i < observations.size(); i++){
        
        //initalize min distance
        double min_dist = 1.00E99;
        //initalize map id
        int update_id = -1;
        //landmark observation measurments
        LandmarkObs o = observations [i];
        
        //for all predictions
        for(int j = 0; j < predicted.size(); j++){
            
            //landmark predicted measurments
            LandmarkObs p = predicted[j];
            //calculate distance between observation and prediction
            double cur_dist = dist(o.x, o.y, p.x, p.y);
            
            //if the current distance < last minimum distance update the minimum distance and id
            if(cur_dist < min_dist){
                min_dist = cur_dist;
                update_id = p.id;
            }
        }
        
        //update observation id with the closest prediction id
        observations[i].id = update_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    //for all particles
    for(int i=0; i < num_particles; i++){
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_cos = cos(particles[i].theta);
        double p_sin = sin(particles[i].theta);
        
        vector<LandmarkObs> trn_obs;
        //for all observations
        for(int j=0; j < observations.size(); j++){
            
            double obs_x = observations[j].x;
            double obs_y = observations[j].y;
            //transform observations to map coordinate system
            double trn_x = p_x + p_cos*obs_x - p_sin*obs_y;
            double trn_y = p_y + p_sin*obs_x + p_cos*obs_y;
            
            trn_obs.push_back(LandmarkObs{observations[j].id,trn_x,trn_y});
        }
        
        vector<LandmarkObs> range_lm;
        //for all landmarks
        for(int j=0; j < map_landmarks.landmark_list.size(); j++){
            
            int lm_id = map_landmarks.landmark_list[j].id_i;
            double lm_x = map_landmarks.landmark_list[j].x_f;
            double lm_y = map_landmarks.landmark_list[j].y_f;
            
            //calculate distance between landmarks
            double lm_dist = dist(lm_x, lm_y, p_x, p_y);
            
            //if landmark is in sensor range
            if(fabs(lm_dist) <= sensor_range){
                range_lm.push_back(LandmarkObs{lm_id,lm_x,lm_y});
            }
        }
        
        //find nearest particle
        dataAssociation(range_lm, trn_obs);

        double weight = 1.0;
        particles[i].weight = 1.0;
        
        //for all transformed observations
        for(int j=0; j < trn_obs.size(); j++){
            double o_id = trn_obs[j].id;
            double o_x = trn_obs[j].x;
            double o_y = trn_obs[j].y;
            
            double pr_x, pr_y;
            //for all landmarks
            for(int k=0; k < range_lm.size(); k++){
                if(range_lm[k].id == o_id){
                    pr_x =  range_lm[k].x;
                    pr_y = range_lm[k].y;
                }
            }
            
            double gauss_norm = (1/(2*M_PI*std_landmark[0]*std_landmark[1]));
            double exponent = (pow((pr_x - o_x),2)/2*pow(std_landmark[0], 2))+(pow((pr_y - o_y),2)/2*pow(std_landmark[1], 2));
            
            //calculate weight
            weight *= gauss_norm * exp(-exponent);
        }
        particles[i].weight = weight;
        weights[i] = weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    //create vector to hold new particles
    vector<Particle> new_particles;
    
    //discrete distribution by weights
    discrete_distribution<int> disc_dist(weights.begin(),weights.end());
    default_random_engine gen;
    
    //for all particles
    for(int i=0; i < particles.size(); i++){
        
        int id = disc_dist(gen);
        Particle p {id,particles[id].x,particles[id].y,particles[id].theta,1.0};
        
        new_particles.push_back(p);
    }
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
    
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
