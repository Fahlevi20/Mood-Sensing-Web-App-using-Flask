// script.js

document.addEventListener("DOMContentLoaded", function () {
    const uploadElement = document.querySelector(".box .upload");
    const textWrapperElement = document.querySelector(".box .text-wrapper");
  
    // Menambahkan event listener untuk tindakan klik
    uploadElement.addEventListener("click", () => {
      // Ganti teks pada .text-wrapper saat elemen diklik
      textWrapperElement.textContent = "Image Uploaded!";
      
      // Ubah warna latar belakang elemen .upload
      uploadElement.style.backgroundColor = "#42f578";
      
      // Anda juga bisa melakukan tindakan lain di sini sesuai kebutuhan.
    });
  });
  