double relativistic(double *q, double *par)
{
  double Q = q[0];
  double gamma = TMath::Sqrt(par[0]*par[0]*(par[0]*par[0]+par[1]*par[1])); //par[0]=M, par[1]=G
  double k = 2*TMath::Sqrt2()*par[0]*par[1]*gamma/TMath::Pi()/TMath::Sqrt(par[0]*par[0]+gamma);
  double d = (Q*Q-par[0]*par[0])*(Q*Q-par[0]*par[0]) + par[0]*par[0]*par[1]*par[1]; 
  return k/d;
}

double nonrelativistic(double *q, double *par)
{
  double Q = q[0];
  return 1./TMath::Pi()/(1 + (Q-par[0])*(Q-par[0])/(par[1]*par[1]/4));
  //return par[1]/2/TMath::Pi()/((Q-par[0])*(Q-par[0]) + par[1]*par[1]/4); //formula to use when G!=2
}

double findratio(double *q, double *par)
{
  double Q = q[0];
  double gamma = TMath::Sqrt(par[0]*par[0]*(par[0]*par[0]+par[1]*par[1])); //par[0]=M, par[1]=G
  double k = 2*TMath::Sqrt2()*par[0]*par[1]*gamma/TMath::Pi()/TMath::Sqrt(par[0]*par[0]+gamma);
  double d = (Q*Q-par[0]*par[0])*(Q*Q-par[0]*par[0]) + par[0]*par[0]*par[1]*par[1]; //relativistic is k/d 
  double n = 1./TMath::Pi()/(1 + (Q-par[0])*(Q-par[0])/(par[1]*par[1]/4)); //non-relativistic
  //double n = par[1]/2/TMath::Pi()/((Q-par[0])*(Q-par[0]) + par[1]*par[1]/4); //formula to use when G!=2
  return k/d/n/10000; //by 10000 to plot together with the BW
}

double one(double *q, double *par)
{
  return 0.0001;
}

void myfunc()
{

   TCanvas* c = new TCanvas("c", "canvas", 800, 800);
   c->SetRealAspectRatio();
   
   auto f1 = new TF1("reltononrel",findratio,0,160,2);
   f1->SetParameters(80,2);
   f1->SetParNames("M","G");
   f1->SetMinimum(0.00001);
   f1->SetMaximum(0.003);
   
   auto f2 = new TF1("rel",relativistic,0,160,2);
   f2->SetParameters(80,2);
   f2->SetParNames("M","G");
      
   auto f3 = new TF1("nonrel",nonrelativistic,0,160,2);
   f3->SetParameters(80,2);
   f3->SetParNames("M","G");

   auto f4 = new TF1("one",one,0,160,2);
   
   f1->Draw();
   f1->SetLineColor(kGreen);

   f2->Draw("SAME");
   f2->SetLineColor(kRed);
   f2->SetLineWidth(2);
   
   f3->Draw("SAME");
   f3->SetLineColor(kBlue);
   f3->SetLineWidth(2);

   f4->Draw("SAME");
   f4->SetLineColor(kBlack);
   f4->SetLineWidth(2);
   f4->SetLineStyle(9);

   auto leg = new TLegend(0.10, 0.68, 0.70, 0.90);
   leg->SetFillStyle(0);
   leg->SetBorderSize(0);
   leg->SetTextSize(0.035);
   leg->SetFillColor(10);
   leg->SetNColumns(1);
   leg->SetHeader("");

   leg->AddEntry(f1, "rel/nonrel*10^-4", "l");
   leg->AddEntry(f2, "rel", "l");
   leg->AddEntry(f3, "nonrel", "l");
   leg->Draw("");
   
   f1->SetTitle("M=80, G=2, BW distributions");

}
