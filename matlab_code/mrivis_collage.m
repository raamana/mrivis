function mrivis_collage( img_spec, fig_handle, rescale_intensity_range, varargin)
% mrivis_collage( img, fig_handle)
%   img  = MR image to be visualized
%   fig_handle = figure handle to display the images
%   scale_intensity_flag - whether to rescale image intensities or not
%       can be a flag ([true]/false)
%       or an array specifying the [ min max ] to which intensities should be rescaled to
%       - This useful for comparison purposes:
%           when images being visualized is exactly what you wanna see and not the rescaled ones
%   varargin: optional text to be displayed

% setting arguments
if nargin < 2
    fig_handle = figure;
end

if nargin < 3
    rescale_intensity_range = true;
end

% reading in data
if ischar(img_spec)
    img_vol = MRIread(img_spec);
    if numel(size(img_vol.vol)) ~= 3
        error('Input volume is not 3D!')
    end
    img = img_vol.vol;
elseif isreal(img_spec)
    if numel(size(img_spec)) ~= 3
        error('Input volume is not 3D!')
    end
    img = img_spec;
else
    display('Invalid input specified. Input either a path to image data, or provide 3d Matrix directly.')
end

% cropping the image to its extents
padding = 5;
[ coords(:,1), coords(:,2), coords(:,3) ] =  ind2sub( size(img), find(img > 0 ) );
beg_coords = max(1        ,min(coords)-padding);
end_coords = min(size(img),max(coords)+padding);
img = img( beg_coords(1):end_coords(1), beg_coords(2):end_coords(2), beg_coords(3):end_coords(3) );

num_cross_sections = 11;
slices = { 
    round( linspace(1,size(img,1),num_cross_sections));...
    round( linspace(1,size(img,2),num_cross_sections));...
    round( linspace(1,size(img,3),num_cross_sections)) 
    };


% by default
RescaleImages = true;

% estimating intensity ranges
if length(rescale_intensity_range) == 1 && rescale_intensity_range
    img_intensity_range = [ min(img(:)), max(img(:)) ];
elseif length(rescale_intensity_range) == 2 % [ min max ]
    img_intensity_range = rescale_intensity_range;
else
    RescaleImages = false;
end

set(0, 'CurrentFigure', fig_handle);
set(fig_handle,'Color','k');

for dim_index =  1 : 3
    for range_index =  2 : num_cross_sections-1 % exluding the end slices
        
        % making the axis
        subplot('Position', get_subplot_pos( dim_index, range_index) );
        
        % getting slice data
        slice = getdim(img,dim_index,slices{dim_index}(range_index) );
        
        % visualizing it
        if RescaleImages
            imagesc(slice, img_intensity_range );
        else
            imshow(slice);
        end

        % adjustments for proper presentation
        colormap gray;
        axis off; 
        axis image;
    end
end

% displaying some annotation text if provided
% good choice would be the location of the input image (for future reference when image is shared or misplaced!)
if nargin > 3
    pos_annot_path_info = [ 0   0.485  1 0.03  ]; % path
    subplot('Position',pos_annot_path_info , 'Color', 'k' ); axis off;
    text(0.05,0.5,varargin{1}, ...
        'Interpreter','none','Color','g', ...
        'BackgroundColor','k','fontsize',12, ...
        'horizontalAlignment','left');
end

end


function pos = get_subplot_pos( dim_index, range_index)
% to identify the positions of the different subplots

designated_base = [ 0 0; 0.5 0.535; 0 0.535];
base = designated_base(dim_index, :);
% bounding box params for a 4-quad 3x3 grid
wBB = 0.16; hBB = 0.155;
%-% Pattern: [ 1 2 3; 4 5 6; 7 8 9]
w = mod(range_index-2, 3)*wBB;
h = abs( floor((range_index-2)/3) - 2 )*hBB;
pos = [ base+[w h] wBB hBB-0.005 ];

end
