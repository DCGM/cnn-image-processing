function results = evalPSNR(origPath, recPath)

OrigImages=dir([origPath '/*.png']);

results = [];
for i = 1:length(OrigImages)
    origName = [origPath '/' OrigImages(i).name];
    recName =  [recPath '/' OrigImages(i).name];

    origImg = imread( origName);
    recImg = imread( recName);
    result = compute_errors(origImg, recImg);
    results = [results; result'];
end

results = mean(results, 1);
fprintf('%s: PSNR: %.2f; PSNR_B: %.2f; BEF: %.2f; SSIM: %.2f; RMSE: %.2f;\n', recPath, results);

end
