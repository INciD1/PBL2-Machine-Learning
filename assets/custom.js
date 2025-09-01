// เพิ่มผลกระทบ fade-in เมื่อโหลดหน้า
$(document).ready(function() {
    // เพิ่มคลาส fade-in ให้กับเนื้อหาหลัก
    $('.dash-tab-content').addClass('fade-in');
    
    // เปลี่ยนแท็บที่เลือกเพื่อเพิ่ม/ลบคลาส fade-in
    $('.dash-tab').on('click', function() {
        $('.dash-tab-content').removeClass('fade-in');
        setTimeout(function() {
            $('.dash-tab-content').addClass('fade-in');
        }, 100);
    });

    // ทำให้การ์ดมีการยกตัวขึ้นเมื่อเมาส์ชี้
    $('.card').hover(
        function() {
            $(this).css('transform', 'translateY(-5px)');
            $(this).css('box-shadow', '0 8px 16px rgba(0,0,0,0.2)');
        },
        function() {
            $(this).css('transform', 'translateY(0)');
            $(this).css('box-shadow', '0 4px 12px rgba(0,0,0,0.08)');
        }
    );
});