$(document).ready(function() {
    // Initialize the bulma carousel for the results section
    var carousels = bulmaCarousel.attach('.carousel', {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: false,
        navigation: true,
        pagination: true,
    });

    // Loop through any carousels and add navigation listeners if needed
    for (var i = 0; i < carousels.length; i++) {
        carousels[i].on('before:show', state => {
            // Optional: pause videos when sliding away
        });
    }

    // Pause all carousel videos except the visible one when slide changes
    // (Optional refinement — videos already muted+loop)
});
