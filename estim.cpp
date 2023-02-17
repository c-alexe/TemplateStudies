
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

double cheb_zeros(double *var, double *par){
   return (TMath::Cos((par[0]-var[0])*TMath::Pi()/par[0]) + par[1])*par[2]; // par[0]=n, par[1]=offset, par[2]=scale, var[0]=m
}

int main()
{  
  double x, y;
  double max_x = 0.8;
  double max_y = 3.5;
  int n_bins_x = 10;
  int n_bins_y = 6;
  double x_bin_width = max_x/n_bins_x;
  double y_bin_width = 2*max_y/n_bins_y;

  // Define the xy weight 
  TF2* wxy = new TF2("w_xy", "[0]*x/TMath::Power(x*x+[1], [2])*[3]/TMath::Sqrt(2*TMath::Pi()*[4])*TMath::Exp(-0.5*(y-[5])*(y-[5])/[4])", 0.0, max_x, -max_y, max_y);
  double sigma2_y = 4.0*4.0;
  wxy->SetParameter(0, 1.0);
  wxy->SetParameter(1, 0.00235);
  wxy->SetParameter(2, 1.0);
  wxy->SetParameter(3, 1.0);
  wxy->SetParameter(4, sigma2_y);
  wxy->SetParameter(5, 0.0);

  /*
  //Toy weight for debugging 
  TF2* w_toy = new TF2("w_toy", "x+y", 0.0, max_x, -max_y, max_y);
  // Fill in histogram bin by bin
  TH2D* hist_toy = new TH2D("hist_toy","weights_toy", n_bins_x, 0., max_x, n_bins_y, -max_y, max_y);
  for(int i=1; i<=n_bins_x; i++){  
    for(int k=1; k<=n_bins_y; k++){
      double weight=w_toy->Eval((i-0.5)*x_bin_width, -max_y + (k-0.5)*y_bin_width); 
      hist_toy->Fill((i-0.5)*x_bin_width, -max_y + (k-0.5)*y_bin_width, weight);  
    }    
  }
  */
  
  // Fill in and normalise histogram
  TH2D* hist = new TH2D("hist","weights", n_bins_x, 0., max_x, n_bins_y, -max_y, max_y);
  hist->FillRandom("w_xy",1000000);
  
  double integral = hist->Integral("width");
  std::cout<<"integral with width is: "<<integral<<"\n";
  hist->Scale(1/integral);
  
  // Roll the histogram, row major
  VectorXd xy_vector(n_bins_x*n_bins_y);
  VectorXd xy_errors(n_bins_x*n_bins_y);
  for(int k=1; k<=n_bins_y; k++){
    for(int i=1; i<=n_bins_x; i++){
      xy_vector((k-1)*n_bins_x+(i-1))=hist->GetBinContent(i,n_bins_y-k+1);
      xy_errors((k-1)*n_bins_x+(i-1))=hist->GetBinErrorLow(i,n_bins_y-k+1);
    }
  }

  //std::cout<<"this is y: "<<xy_vector;

  // Define J matrix from lambda functions
  Eigen::MatrixXd J(n_bins_x*n_bins_y,(n_bins_x+1)*(n_bins_y+1));

  TF1* cheb_x = new TF1("cheb_x", cheb_fct, 0, max_x, 4); 
     cheb_x->SetParNames("n","offset","scale","m");
     cheb_x->SetParameter("n", n_bins_x);
     cheb_x->SetParameter("offset", 1.0);
     cheb_x->SetParameter("scale", 0.5*max_x);

  TF1* cheb_y = new TF1("cheb_y", cheb_fct, -max_y, max_y, 4);
     cheb_y->SetParNames("n","offset","scale","m");
     cheb_y->SetParameter("n", n_bins_y);
     cheb_y->SetParameter("offset", 0.0);
     cheb_y->SetParameter("scale", max_y);
     
  for(int h=0; h<n_bins_x*n_bins_y; h++){
     VectorXd J_vector((n_bins_x+1)*(n_bins_y+1));
     int q=h/n_bins_x;
     int r=h%n_bins_x;
     for(int j=0; j<=n_bins_y; j++){
        cheb_y->SetParameter("m", j);
        for(int k=0; k<=n_bins_x; k++){
           cheb_x->SetParameter("m", k);
           J_vector(j*(n_bins_x+1)+k)=cheb_x->Integral(r*x_bin_width,(r+1)*x_bin_width,1.e-10)*cheb_y->Integral(y_bin_width*(n_bins_y-q-1),y_bin_width*(n_bins_y-q),1.e-10); 
        }
     }
     
     J.row(h)=J_vector;
  }

  //std::cout<<"this is J:"<<J;
  
  // Write V^-1/2
  Eigen::MatrixXd V(n_bins_x*n_bins_y, n_bins_x*n_bins_y);
  V=MatrixXd::Zero(n_bins_x*n_bins_y, n_bins_x*n_bins_y);
  for(int i=0; i<n_bins_x*n_bins_y; i++){
    V(i,i)=1/xy_errors(i);
  }

  //std::cout << "Here is V^-1/2:"<<"\n" << V << std::endl;

  // Solve for f
  Eigen::MatrixXd A = V*J;
  Eigen::MatrixXd b = V*xy_vector;
  Eigen::VectorXd f_estim = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

  //std::cout << "Here is f_estim:"<<"\n" << f_estim << std::endl;

  // Unroll f
  Eigen::MatrixXd f_unrolled(n_bins_y+1,n_bins_x+1);
  for(int j=0; j<=n_bins_y; j++){
    for(int k=0; k<=n_bins_x; k++){
      f_unrolled(j,k)=f_estim(j*(n_bins_x+1)+k);
    }
  }
  std::cout << "Here is f_estim unrolled:"<<"\n" << f_unrolled << "\n";
  
  // f in terms of cheb zeros
  // Fill in histogram bin by bin
  TH2D* hist_f_estim = new TH2D("hist_f_estim","f_estim", n_bins_x+1, 0., max_x, n_bins_y+1, -max_y, max_y);

  TF1* cheb_zx = new TF1("cheb_zx", cheb_zeros, 0, max_x, 3);
     cheb_zx->SetParNames("n","offset","scale");
     cheb_zx->SetParameter("n", n_bins_x);
     cheb_zx->SetParameter("offset", 1.0);
     cheb_zx->SetParameter("scale", 0.5*max_x);

  TF1* cheb_zy = new TF1("cheb_zy", cheb_zeros, -max_y, max_y, 3);
      cheb_zy->SetParNames("n","offset","scale");
      cheb_zy->SetParameter("n", n_bins_y);
      cheb_zy->SetParameter("offset", 0.0);
      cheb_zy->SetParameter("scale", max_y);

  Eigen::MatrixXd compare(n_bins_y+1,n_bins_x+1);
  Eigen::MatrixXd zerosx(n_bins_y+1,n_bins_x+1);
  Eigen::MatrixXd zerosy(n_bins_y+1,n_bins_x+1);
  for(int j=0; j<=n_bins_y; j++){
    for(int k=0; k<=n_bins_x; k++){
      zerosx(n_bins_y-j,k)=cheb_zx->Eval(k);
      zerosy(n_bins_y-j,k)=cheb_zy->Eval(j);
      compare(j,k)=wxy->Eval(cheb_zx->Eval(k), cheb_zy->Eval(j));
      hist_f_estim->Fill(cheb_zx->Eval(k), cheb_zy->Eval(j), f_unrolled(j,k)); //not useful in this form
    }
  }
  //std::cout<<"zerosx:"<<"\n"<<zerosx;
  //std::cout<<"zerosy:"<<"\n"<<zerosy;

  double integral_estim = hist_f_estim->Integral("width");
  //std::cout<<"integral estim with width is: "<<integral_estim<<"\n";
  //hist_f_estim->Scale(1/integral_estim);

  std::cout<<"w_xy(zerox,zeroy):"<<"\n"<<compare<<"\n";

  Eigen::MatrixXd ratios(n_bins_y+1,n_bins_x+1);
  for(int j=0; j<=n_bins_y; j++){
    for(int k=0; k<=n_bins_x; k++){
      ratios(j,k)=compare(j,k)/f_unrolled(j,k);
    }
  }
  std::cout<<"w_xy(zerox,zeroy)/f_estim:"<<"\n"<<ratios<<"\n";
  
  //Print the values of the "data" hist bins

  Eigen::MatrixXd hist_values(n_bins_y,n_bins_x);
  for(int i=1; i<=n_bins_x; i++){
    for(int k=1; k<=n_bins_y; k++){
      double x_bin_width = max_x/n_bins_x;
      double y_bin_width = 2*max_y/n_bins_y;
      hist_values(k-1,i-1)=hist->GetBinContent(i,k);
      //std::cout<<"for i="<<i<<", k="<<k<<": hist get value="<<hist->GetBinContent(i,k)<<", error vect["<<(k-1)*n_bins_x+(i-1)<<"]: "<<xy_errors((k-1)*n_bins_x+(i-1))<<", for  x>"<<(i-0.5)*x_bin_width<<", y>"<<-max_y + (k-0.5)*y_bin_width<<"\n";
    }
  }
  //std::cout<<"'Data' hist values: "<<"\n"<<hist_values<<"\n";

  // Plot erorrs
  TH1D* err = new TH1D("err","errors", n_bins_x*n_bins_y, 0, n_bins_x*n_bins_y);
  for(int i=1; i<=n_bins_x; i++){
    for(int k=1; k<=n_bins_y; k++){
      err->Fill((k-1)*n_bins_x+(i-1), xy_errors((k-1)*n_bins_x+(i-1)));
    }
  }
  
  TCanvas *c1 = new TCanvas("c1", "weights", 600, 600);
  c1->cd();
  //gStyle->SetPalette(87);
  gStyle->SetNumberContours(256);
  hist->Draw("COLZ");
  c1->SaveAs("hist.pdf");
  hist_f_estim->SetStats(0);
  hist_f_estim->Draw("COLZ");
  c1->SaveAs("hist_f_estim.pdf");
  err->Draw("HIST");
  c1->SaveAs("err.pdf");
}


