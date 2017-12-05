function matrix = getdim( A, dimToSelect, sliceNumber, varargin )
% dynamic selection of dimension
%
% useful to select axial, coronal and sagittal slices from 3D matrix
% dynamically..

ss=repmat({':'},1,ndims(A));
ss{dimToSelect}= sliceNumber ;
% matrix = subsref(A,struct('type','()','subs',ss));
matrix = squeeze( subsref(A,substruct('()',ss)) );


% the following is not done by default out to maintain correspondence when
% overlaying with a segmentation
FlipAxialSliceFlag = false;
if nargin > 3
    FlipAxialSliceFlag = varargin{1};
end

% % this works only when visualizing MRI (or any scalar-valued images)
% % to get the axial view displayed properly...
if FlipAxialSliceFlag && ismatrix(matrix) && dimToSelect == 1; 
    matrix=flipud(matrix'); 
end


