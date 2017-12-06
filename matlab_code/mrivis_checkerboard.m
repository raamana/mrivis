function mrivis_checkerboard( img_spec1, img_spec2, patch_size,...
    fig_handle, rescale_intensity_range, varargin)
% mrivis_collage( img_spec1, img_spec2, fig_handle)
%   img_spec1  = MR image (or path to one) to be visualized
%   img_spec2  = second MR image/path to be compared
%   patch_size = size of patch in checkerboard (default 10 voxels square)
%       This could be rectangular also e.g. [10, 30] specifying width and height of patch.
%   fig_handle = figure handle to display the images
%   scale_intensity_flag - whether to rescale image intensities or not
%       can be a flag ([true]/false)
%       or an array specifying the [ min max ] to which intensities should be rescaled to
%       - This useful for comparison purposes:
%           when images being visualized is exactly what you wanna see and not the rescaled ones
%   varargin: optional text to be displayed

% setting arguments
if nargin < 3
    patch_size = [10, 10];
end

if nargin < 4
    fig_handle = figure;
end

if nargin < 5
    rescale_intensity_range = true;
end

img1 = get_image(img_spec1);
img2 = get_image(img_spec2);

img_size = size(img1);
img_size2 = size(img2);
if ~isequal(img_size,img_size2)
    error('size mismatch! Two images to be compared must be of the same size in all dimensions.')
end

% cropping the images to their extents
padding = 5;
[img1, img2] = crop_to_extents(img1, img2, padding);

cropped_img_size = size(img1);

num_cross_sections_total = 24;
% skipping first and last 3
num_cs_to_skip = 6;
slices = {
    round( linspace(1,cropped_img_size(1),num_cross_sections_total));...
    round( linspace(1,cropped_img_size(2),num_cross_sections_total));...
    round( linspace(1,cropped_img_size(3),num_cross_sections_total))
    };


RescaleImages = true; % by default

% estimating intensity ranges
if length(rescale_intensity_range) == 1 && rescale_intensity_range
    img_intensity_range = [ min(img1(:)), max(img1(:)) ];
elseif length(rescale_intensity_range) == 2 % [ min max ]
    img_intensity_range = rescale_intensity_range;
else
    RescaleImages = false;
end

if numel(unique(img_intensity_range))==1
    RescaleImages = false;
end

set(0, 'CurrentFigure', fig_handle);
set(fig_handle,'Color','k');

for dim_index =  1 : 3
    slices_this_dim = slices{dim_index}(num_cs_to_skip+1:end-num_cs_to_skip);
    for range_index =  1 : length(slices_this_dim)

        % making the axis
        subplot('Position', get_subplot_pos( dim_index, range_index) );

        % getting slice data
        slice1 = getdim(img1, dim_index, slices_this_dim(range_index) );
        slice2 = getdim(img2, dim_index, slices_this_dim(range_index) );

        % making a mask for checkers
        checkers = get_checkers(size(slice1), patch_size);
        mixed = mix_slices(slice1, slice2, checkers);

        % visualizing it
        if RescaleImages
            imagesc(mixed, img_intensity_range );
        else
            imshow(mixed);
        end

        % adjustments for proper presentation
        colormap gray;
        axis off;
        axis image;
    end
end

% displaying some annotation text if provided
% good choice would be the location of the input image (for future refwhen image is shared or misplaced!)
if nargin > 5
    pos_annot_path_info = [ 0   0.01  1 0.03  ]; % path
    subplot('Position',pos_annot_path_info , 'Color', 'k' ); axis off;
    text(0.05,0.5,varargin{1}, ...
        'Interpreter','none','Color','g', ...
        'BackgroundColor','k','fontsize',12, ...
        'horizontalAlignment','left');
end

end

function img = get_image(img_spec)
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
    display('Invalid input specified!')
    display('Input either a path to image data, or provide 3d Matrix directly.')
end

end

function checkers = get_checkers(slice_size, patch_size)
% creates checkerboard of a given tile size, filling a given slice

black = zeros(patch_size);
white = ones(patch_size);
tile = [black white; white black];
tile_size = size(tile);

% using ceil so we can clip the extra portions
num_repeats = ceil(slice_size ./ tile_size);
checkers = repmat(tile, num_repeats);

collage_size = size(checkers);
if any(collage_size>slice_size)
    if collage_size(1) > slice_size(1)
        checkers(slice_size(1)+1:end,:) = [];
    end
    if collage_size(2) > slice_size(2)
        checkers(:, slice_size(2)+1:end) = [];
    end
end
% patch_size, tile_size, slice_size, num_repeats, collage_size, size(checkers)

end

function mixed = mix_slices(slice1, slice2, checkers)

mixed = slice1;
mixed(checkers>0) = slice2(checkers>0);

end

function [img1, img2] = crop_to_extents(img1, img2, padding)

[beg_coords1, end_coords1] = crop_coords(img1, padding);
[beg_coords2, end_coords2] = crop_coords(img2, padding);

beg_coords = min(beg_coords1, beg_coords2);
end_coords = max(end_coords1, end_coords2);

img1 = crop_3dimage(img1, beg_coords, end_coords);
img2 = crop_3dimage(img2, beg_coords, end_coords);

end

function [beg_coords, end_coords] = crop_coords(img, padding)

[ coords(:,1), coords(:,2), coords(:,3) ] =  ind2sub( size(img), find(img > 0 ) );
if isempty(coords)
    end_coords = size(img);
    beg_coords = ones(1, length(size(img)));
else
    beg_coords = max(1        , min(coords)-padding);
    end_coords = min(size(img), max(coords)+padding);
end

end


function cropped_img = crop_3dimage(img,beg_coords,end_coords)

cropped_img = img(...
    beg_coords(1):end_coords(1), ...
    beg_coords(2):end_coords(2), ...
    beg_coords(3):end_coords(3) );

end


function pos = get_subplot_pos( dim_index, range_index)
% to identify the positions of the different subplots

if range_index <= 6
    designated_base = [ 0 0.00; 0 0.33; 0 0.66 ];
else
    designated_base = [ 0 0.16; 0 0.49; 0 0.825];
end

base = designated_base(dim_index, :);

% bounding box (BB) params for a sequential 6x1 grid
wBB = 0.16;
hBB = 0.155;

%-% Pattern: [ 1 2 3 4 5 6]
w = mod(range_index-1, 6)*wBB;
h = 0;
pos = [ base+[w h] wBB hBB-0.005 ];

end

function pos = get_subplot_pos_quad( dim_index, range_index)
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
