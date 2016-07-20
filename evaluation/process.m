% Here comes the path to dir of original files
OrigPathDir='Live1/orig';
% Path to dirs with reconstructions
ReconstructionPathDir='Live1/L05-psnr';

OrigImages=dir([OrigPathDir '/*.png']);
ReconstructionDirs=dir(ReconstructionPathDir);

for i_ReconstructionDirs = 3:size(ReconstructionDirs,1)

  %fprintf('%s\n', ReconstructionDirs(i_ReconstructionDirs).name);
  % Works in Octave:  dir([ReconstructionPathDir '/' ReconstructionDirs(i_ReconstructionDirs).name '/' '*[png,jpg]']);
  ReconstructionImages=[ dir(fullfile([ReconstructionPathDir '/' ReconstructionDirs(i_ReconstructionDirs).name], '*.jpg')) ; dir(fullfile([ReconstructionPathDir '/' ReconstructionDirs(i_ReconstructionDirs).name], '*.png'))];
  results=[];
   
  for i = 1:size(ReconstructionImages,1)
    %origImg = imread([OrigPathDir '/' OrigImages(i).name]);
    [s,e] = regexp(ReconstructionImages(i).name, '^.*?jpg');
    origName = dir([OrigPathDir '/' ReconstructionImages(i).name(s:e-3) '*'] );
    origImg = imread([OrigPathDir '/' origName(1).name]);
    reconstructedImg = imread( [ReconstructionPathDir '/' ReconstructionDirs(i_ReconstructionDirs).name '/' ReconstructionImages(i).name]);
    result = compute_errors(origImg, reconstructedImg);
    results=[results; result'];
    fprintf('PSNR: %f, Orig: %s, Reconstruction: %s \n', result(1), origName(1).name, ReconstructionImages(i).name)
  
  end

  ReconstructionDirs(i_ReconstructionDirs).results=results;
  ReconstructionDirs(i_ReconstructionDirs).results_mean=mean(results);
  
  fprintf('%s: PSNR: %f; PSNR_B: %f; BEF: %f; SSIM: %f; RMSE: %f;\n', ReconstructionDirs(i_ReconstructionDirs).name, ReconstructionDirs(i_ReconstructionDirs).results_mean);
 end
