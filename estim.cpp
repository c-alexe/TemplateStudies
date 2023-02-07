
#include <ROOT/RDataFrame.hxx>
#include "TFile.h"
#include "TRandom3.h"
#include "TVector.h"
#include "TVectorT.h"
#include "TMath.h"
#include "TF1.h"
#include "TF2.h"
#include "TStyle.h"
#include "TCanvas.h"
#include <TMatrixD.h>
#include <TStopwatch.h>
#include <ROOT/RVec.hxx>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;
using namespace ROOT;
typedef ROOT::VecOps::RVec<double> RVecD;
using ROOT::RDF::RNode; 

double cheb_fct(double *var, double *par){
  double x=var[0];
  double den = 0.;
  double num = 0.;
  for(int i = 0; i <= par[0] ; i++){ //par[0]=n
    int sign = i%2==0 ? +1 :-1;
    double xj = (TMath::Cos((par[0]-i)*TMath::Pi()/par[0]) + par[1])*par[2]; //par[1]=offset, par[2]=scale
    if(x==xj) return 1.0;// protect from nan      
    double val = sign/(x-xj);
    if(i==0 || i==par[0]) val *= 0.5;
    den += val;
    if(i==par[3]) num = val; //par[3]=m
  }                                             
  return num/den;
}

int main()
{  
  double x, y;
  double max_x = 0.8;
  double max_y = 3.5;
  int n_bins_x = 10;
  int n_bins_y = 6; 

  // Define the xy weight 
  TF2* wxy = new TF2("w_xy", "[0]*x/TMath::Power(x*x+[1], [2])*[3]/TMath::Sqrt(2*TMath::Pi()*[4])*TMath::Exp(-0.5*(y-[5])*(y-[5])/[4])", 0.0, max_x, -max_y, max_y);
  double sigma2_y = 4.0*4.0;
  wxy->SetParameter(0, 1.0);
  wxy->SetParameter(1, 0.00235);
  wxy->SetParameter(2, 1.0);
  wxy->SetParameter(3, 1.0);
  wxy->SetParameter(4, sigma2_y);
  wxy->SetParameter(5, 0.0);

  /*  // Fill in histogram bin by bin
  TH2D* hist = new TH2D("hist","weights", n_bins_x, 0., max_x, n_bins_y, -max_y, max_y);
  for(int i=1; i<=n_bins_x; i++){  
    for(int k=1; k<=n_bins_y; k++){
      double x_bin_width = max_x/n_bins_x;
      double y_bin_width = 2*max_y/n_bins_y;
      double weight=wxy->Eval((i-0.5)*x_bin_width, -max_y + (k-0.5)*y_bin_width); 
      hist_notnorm->Fill((i-0.5)*x_bin_width, -max_y + (k-0.5)*y_bin_width, weight);  
    }    
  }
   */

  // Fill in and normalise histogram
  TH2D* hist = new TH2D("hist","weights", n_bins_x, 0., max_x, n_bins_y, -max_y, max_y);
  hist->FillRandom("w_xy",1000000);
  
  double integral = hist->Integral("width");
  std::cout<<"integral with width is: "<<integral<<"\n";
  hist->Scale(1/integral);
  
  // Store the errors into vector
  VectorXd xy_errors(n_bins_x*n_bins_y);
  for(int i=1; i<=n_bins_x; i++){
    for(int k=1; k<=n_bins_y; k++){
      xy_errors((k-1)*n_bins_x+(i-1))=hist->GetBinErrorLow(i,k);
    }
  }

  // Define lambda functions

  TF1* cheb = new TF1("cheb", cheb_fct, 0, max_x, 4);
  cheb->SetParameters(n_bins_x, 0.5*max_x, 1.0, 5);
  cheb->SetParNames("n","offset","scale","m");
  
  // Plot erorrs
  TH1D* err = new TH1D("err","errors", n_bins_x*n_bins_y, 0, n_bins_x*n_bins_y);
  for(int i=1; i<=n_bins_x; i++){
    for(int k=1; k<=n_bins_y; k++){
      err->Fill((k-1)*n_bins_x+(i-1), xy_errors((k-1)*n_bins_x+(i-1)));
    }
  }
  
//    /* Print the values of the bins

  for(int i=1; i<=n_bins_x; i++){
    for(int k=1; k<=n_bins_y; k++){
      double x_bin_width = max_x/n_bins_x;
      double y_bin_width = 2*max_y/n_bins_y;
      std::cout<<"for i="<<i<<", k="<<k<<": hist get value="<<hist->GetBinContent(i,k)<<", error vect["<<(k-1)*n_bins_x+(i-1)<<"]: "<<xy_errors((k-1)*n_bins_x+(i-1))<<", for  x>"<<(i-0.5)*x_bin_width<<", y>"<<-max_y + (k-0.5)*y_bin_width<<"\n";
    }
  }

//  */
  
  TCanvas *c1 = new TCanvas("c1", "weights", 600, 600);
  c1->cd();
  //gStyle->SetPalette(87);
  gStyle->SetNumberContours(256);
  hist->Draw("COLZ");
  c1->SaveAs("hist.pdf");
  err->Draw("HIST");
  c1->SaveAs("err.pdf");
  cheb->Draw();
  c1->SaveAs("cheb.pdf");
}


