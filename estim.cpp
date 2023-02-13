
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
  for(int i = 0; i <= par[0] ; i++){ // par[0]=n
    int sign = i%2==0 ? +1 :-1;
    double xj = (TMath::Cos((par[0]-i)*TMath::Pi()/par[0]) + par[1])*par[2]; // par[1]=offset, par[2]=scale
    if(x==xj) return 1.0;// protect from nan      
    double val = sign/(x-xj);
    if(i==0 || i==par[0]) val *= 0.5;
    den += val;
    if(i==par[3]) num = val; // par[3]=m
  }                                             
  return num/den;
}

double cheb_zeros(double *par){
   return (TMath::Cos((par[0]-par[3])*TMath::Pi()/par[0]) + par[1])*par[2]; // par[1]=offset, par[2]=scale, par[3]=j
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

  // Define J^T matrix from lambda functions

  Eigen::MatrixXd JT(n_bins_x, n_bins_y);
  
  TF1* cheb_x = new TF1("cheb_x", cheb_fct, 0, max_x, 4); 
  cheb_x->SetParNames("n","offset","scale","m");
  cheb_x->SetParameter("n", n_bins_x);
  cheb_x->SetParameter("offset", 1.0);
  cheb_x->SetParameter("scale", 0.5*max_x);
  for(int j=0; j<n_bins_x; j++){
  cheb_x->SetParameter("m", j);
     for(int k=0; k<n_bins_y; k++){
        JT(j,k)=cheb_x->Integral(0,max_x,1.e-10);
     }
  }

  // std::cout << "Here is the matrix JT:\n" << JT << std::endl;
  
  TF1* cheb_y = new TF1("cheb_y", cheb_fct, -max_y, max_y, 4);
  cheb_y->SetParNames("n","offset","scale","m");
  cheb_y->SetParameter("n", n_bins_y);
  cheb_y->SetParameter("offset", 0.0);
  cheb_y->SetParameter("scale", max_y);
  for(int k=0; k<n_bins_y; k++){
  cheb_y->SetParameter("m", k);
     for(int j=0; j<n_bins_x; j++){
       JT(j,k)=JT(j,k)*cheb_y->Integral(-max_y,max_y,1.e-10);
       // std::cout<<cheb_y->Integral(-max_y,max_y,1.e-10)<<" ";
     }
  }

  std::cout << "Here is the matrix JT:\n" << JT << std::endl;

  // Define f
  TH2D* func_cheb = new TH2D("func_cheb","func_cheb", n_bins_x, 0., max_x, n_bins_y, -max_y, max_y);
  for(int i=1; i<=n_bins_x; i++){
    for(int k=1; k<=n_bins_y; k++){
      double x_bin_width = max_x/n_bins_x;
      double y_bin_width = 2*max_y/n_bins_y;
      double weight=wxy->Eval((i-0.5)*x_bin_width, -max_y + (k-0.5)*y_bin_width);
      hist_notnorm->Fill((i-0.5)*x_bin_width, -max_y + (k-0.5)*y_bin_width, weight);
    }
  }
  
  
  // Define f, nono this f should be a matrix of the weights function evaluated at different zeros of the cheb fct
  Eigen::MatrixXd f(n_bins_y, n_bins_x);
  for(int i=0; i<n_bins_y; i++){
     for(int k=0; k<n_bins_x; k++){
       f(i,k)=hist->GetBinContent(k+1,i+1); // watch order of indices for matrix vs histo
     }
  }
    
  // std::cout << "Here is the matrix f:\n" << f << std::endl;

  Eigen::MatrixXd JTf(n_bins_x, n_bins_y);
  JTf=JT*f;

  VectorXd JTf_vector(n_bins_x*n_bins_y);
  for(int i=0; i<n_bins_x; i++){
    for(int k=0; k<n_bins_y; k++){
      JTf_vector(k*n_bins_x+i)=JTf(k,i);
    }
  }

  VectorXd JT_vector(n_bins_x*n_bins_y);
  for(int i=0; i<n_bins_y; i++){
    for(int k=0; k<n_bins_x; k++){
      JT_vector(k*n_bins_y+i)=JT(k,i);
    }
  }

  std::cout << "Here is JT:\n" << JT_vector << std::endl;
  
  std::cout << "Here is their product JT*f:\n" << JTf << std::endl;
  //std::cout << "Here is their product JT*f:\n" << JTf_vector << std::endl;

  // Write V^-1/2
  Eigen::MatrixXd V(n_bins_x*n_bins_y, n_bins_x*n_bins_y);
  V=MatrixXd::Zero(n_bins_x*n_bins_y, n_bins_x*n_bins_y);
  for(int i=0; i<n_bins_x*n_bins_y; i++){
    V(i,i)=1/xy_errors(i);
  }

  //std::cout << "Here is V^-1/2:\n" << V << std::endl;

  // Solve for f
  Eigen::MatrixXd A = V*JT_vector;
  Eigen::MatrixXd b = V*xy_errors;
  Eigen::VectorXd f_estim = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

  std::cout << "Here is f_estim:\n" << f_estim << std::endl;
  
  // Plot erorrs
  TH1D* err = new TH1D("err","errors", n_bins_x*n_bins_y, 0, n_bins_x*n_bins_y);
  for(int i=1; i<=n_bins_x; i++){
    for(int k=1; k<=n_bins_y; k++){
      err->Fill((k-1)*n_bins_x+(i-1), xy_errors((k-1)*n_bins_x+(i-1)));
    }
  }
  
  /* Print the values of the bins

  for(int i=1; i<=n_bins_x; i++){
    for(int k=1; k<=n_bins_y; k++){
      double x_bin_width = max_x/n_bins_x;
      double y_bin_width = 2*max_y/n_bins_y;
      std::cout<<"for i="<<i<<", k="<<k<<": hist get value="<<hist->GetBinContent(i,k)<<", error vect["<<(k-1)*n_bins_x+(i-1)<<"]: "<<xy_errors((k-1)*n_bins_x+(i-1))<<", for  x>"<<(i-0.5)*x_bin_width<<", y>"<<-max_y + (k-0.5)*y_bin_width<<"\n";
    }
  }

  */
  
  TCanvas *c1 = new TCanvas("c1", "weights", 600, 600);
  c1->cd();
  //gStyle->SetPalette(87);
  gStyle->SetNumberContours(256);
  hist->Draw("COLZ");
  c1->SaveAs("hist.pdf");
  err->Draw("HIST");
  c1->SaveAs("err.pdf");
  cheb_y->Draw();
  c1->SaveAs("cheb.pdf");
}


